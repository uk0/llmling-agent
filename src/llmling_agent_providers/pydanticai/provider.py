"""Agent provider implementations."""

from __future__ import annotations

from contextlib import asynccontextmanager
from functools import wraps
import inspect
from typing import TYPE_CHECKING, Any, cast, get_args

from llmling import ToolError
from llmling_models import AllModels, infer_model
import logfire
from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.result import StreamedRunResult

from llmling_agent.common_types import ModelProtocol
from llmling_agent.log import get_logger
from llmling_agent.models.content import BaseContent
from llmling_agent.models.context import AgentContext
from llmling_agent.tasks.exceptions import (
    ChainAbortedError,
    RunAbortedError,
    ToolSkippedError,
)
from llmling_agent.utils.inspection import has_argument_type
from llmling_agent_providers.base import AgentProvider, ProviderResponse
from llmling_agent_providers.pydanticai.utils import (
    content_to_model_message,
    format_part,
    get_tool_calls,
)


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

    from pydantic_ai.agent import EndStrategy
    from pydantic_ai.tools import RunContext

    from llmling_agent.agent.conversation import ConversationManager
    from llmling_agent.common_types import ModelType
    from llmling_agent.models.content import Content
    from llmling_agent.tools.base import ToolInfo


logger = get_logger(__name__)


class PydanticAIProvider(AgentProvider):
    """Provider using pydantic-ai as backend."""

    _conversation: ConversationManager

    def __init__(
        self,
        *,
        model: str | ModelProtocol | None = None,
        name: str = "agent",
        retries: int = 1,
        result_retries: int | None = None,
        end_strategy: EndStrategy = "early",
        defer_model_check: bool = False,
        debug: bool = False,
        **kwargs: Any,
    ):
        """Initialize pydantic-ai backend.

        Args:
            model: Model to use for responses
            name: Agent name
            retries: Number of retries for failed operations
            result_retries: Max retries for result validation
            end_strategy: How to handle tool calls with final result
            defer_model_check: Whether to defer model validation
            debug: Whether to enable debug mode
            kwargs: Additional arguments for PydanticAI agent
        """
        super().__init__(model=model)
        self._debug = debug
        self._kwargs = dict(
            model=model,
            name=name,
            tools=[],
            retries=retries,
            end_strategy=end_strategy,
            result_retries=result_retries,
            defer_model_check=defer_model_check,
            deps_type=AgentContext,
            **kwargs,
        )

    async def get_model_names(self) -> list[str]:
        """Get list of all known model names."""
        return list(get_args(KnownModelName)) + list(get_args(AllModels))

    def get_agent(self, system_prompt: str) -> PydanticAgent[Any, Any]:
        agent = PydanticAgent(system_prompt=system_prompt, **self._kwargs)  # type: ignore
        tools = [t for t in self.tool_manager.values() if t.enabled]
        for tool in tools:
            wrapped = (
                self.wrap_tool(tool, self._context)
                if self._context
                else tool.callable.callable
            )
            if has_argument_type(wrapped, "RunContext"):
                agent.tool(wrapped)
            else:
                agent.tool_plain(wrapped)
        return agent

    def __repr__(self) -> str:
        model = f", model={self.model_name}" if self.model_name else ""
        return f"PydanticAI({self.name!r}{model})"

    @property
    def model(self) -> str | ModelType:
        return self._model

    def wrap_tool(
        self,
        tool: ToolInfo,
        agent_ctx: AgentContext,
    ) -> Callable[..., Awaitable[Any]]:
        """Wrap tool with confirmation handling.

        Current situation is: We only get all infos for tool calls for functions with
        RunContext. In order to migitate this, we "fallback" to the AgentContext, which
        at least provides some information.
        """
        original_tool = tool.callable.callable

        @wraps(original_tool)
        async def wrapped_with_ctx(ctx: RunContext[AgentContext], *args, **kwargs):
            result = await agent_ctx.handle_confirmation(ctx, tool, kwargs)
            match result:
                case "allow":
                    if inspect.iscoroutinefunction(original_tool):
                        return await original_tool(ctx, *args, **kwargs)
                    return original_tool(ctx, *args, **kwargs)
                case "skip":
                    msg = f"Tool {tool.name} execution skipped"
                    raise ToolSkippedError(msg)
                case "abort_run":
                    msg = "Run aborted by user"
                    raise RunAbortedError(msg)
                case "abort_chain":
                    msg = "Agent chain aborted by user"
                    raise ChainAbortedError(msg)

        @wraps(original_tool)
        async def wrapped_without_ctx(*args, **kwargs):
            result = await agent_ctx.handle_confirmation(agent_ctx, tool, kwargs)
            match result:
                case "allow":
                    if inspect.iscoroutinefunction(original_tool):
                        return await original_tool(*args, **kwargs)
                    return original_tool(*args, **kwargs)
                case "skip":
                    msg = f"Tool {tool.name} execution skipped"
                    raise ToolError(msg)
                case "abort_run":
                    msg = "Run aborted by user"
                    raise ToolError(msg)
                case "abort_chain":
                    msg = "Agent chain aborted by user"
                    raise ToolError(msg)

        return (
            wrapped_with_ctx
            if has_argument_type(tool.callable.callable, "RunContext")
            else wrapped_without_ctx
        )

    @logfire.instrument("Pydantic-AI call. model: {model} result type {result_type}.")
    async def generate_response(
        self,
        *prompts: str | Content,
        message_id: str,
        result_type: type[Any] | None = None,
        model: ModelType = None,
        store_history: bool = True,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Generate response using pydantic-ai.

        Args:
            prompts: Texts / Image contents to respond to
            message_id: ID to assign to the response and tool calls
            result_type: Optional type for structured responses
            model: Optional model override for this call
            store_history: Whether the message exchange should be added to the
                           context window
            system_prompt: System prompt from SystemPrompts manager
            kwargs: Additional arguments for pydantic-ai (unused)

        Returns:
            Response message with optional structured content
        """
        agent = self.get_agent(system_prompt or "")
        message_history = self.conversation.get_history()
        use_model = model or self.model
        if isinstance(use_model, str):
            use_model = infer_model(use_model)
            self.model_changed.emit(use_model)
        try:
            text_prompts = [p for p in prompts if isinstance(p, str)]
            content_prompts = [p for p in prompts if isinstance(p, BaseContent)]

            # Get normal text prompt
            prompt = await self.format_prompts(text_prompts)

            # Convert Content objects to ModelMessages
            if content_prompts:
                content_message = content_to_model_message(tuple(content_prompts))
                message_history = [*message_history, content_message]

            # Run with complete history
            result = await agent.run(
                prompt,
                deps=self._context,  # type: ignore
                message_history=message_history,
                model=model or self.model,  # type: ignore
                result_type=result_type or str,
            )

            # Extract tool calls and set message_id
            new_msgs = result.new_messages()
            tool_calls = get_tool_calls(new_msgs, dict(self.tool_manager._items))
            for call in tool_calls:
                call.message_id = message_id
                call.context_data = self._context.data if self._context else None
            if store_history:
                self.conversation._last_messages = list(new_msgs)
                self.conversation.set_history(result.all_messages())
            resolved_model = (
                use_model.name() if isinstance(use_model, Model) else str(use_model)
            )
            return ProviderResponse(
                content=result.data,
                tool_calls=tool_calls,
                usage=result.usage(),
                model_name=resolved_model,
            )
        finally:
            # Restore original model in signal if we had an override
            if model:
                original = self.model
                if isinstance(original, str):
                    original = infer_model(original)
                self.model_changed.emit(original)

    @property
    def name(self) -> str:
        """Get agent name."""
        return self._kwargs.get("name", "agent")  # type: ignore

    @name.setter
    def name(self, value: str | None):
        """Set agent name."""
        self._kwargs["name"] = value

    def set_model(self, model: ModelType):
        """Set the model for this agent.

        Args:
            model: New model to use (name or instance)

        Emits:
            model_changed signal with the new model
        """
        old_name = self.model_name
        if isinstance(model, str):
            model = infer_model(model)
        self._model = model
        self._kwargs["model"] = model
        self.model_changed.emit(model)
        logger.debug("Changed model from %s to %s", old_name, self.model_name)

    @property
    def model_name(self) -> str | None:
        """Get the model name in a consistent format."""
        match model := self._kwargs["model"]:
            case str() | None:
                return model
            case ModelProtocol():
                return model.name()
            case _:
                msg = f"Invalid model type: {model}"
                raise ValueError(msg)

    @asynccontextmanager
    async def stream_response(
        self,
        *prompts: str | Content,
        message_id: str,
        result_type: type[Any] | None = None,
        model: ModelType = None,
        store_history: bool = True,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamedRunResult]:  # type: ignore[type-var]
        """Stream response using pydantic-ai."""
        message_history = self.conversation.get_history()
        agent = self.get_agent(system_prompt or "")
        use_model = model or self.model
        if isinstance(use_model, str):
            use_model = infer_model(use_model)

        if model:
            self.model_changed.emit(use_model)

        if self._debug:
            from devtools import debug

            debug(agent)
        prompt = await self.format_prompts(prompts)
        async with agent.run_stream(
            prompt,
            deps=self._context,  # type: ignore
            message_history=message_history,
            model=model or self.model,  # type: ignore
            result_type=result_type or str,
        ) as stream_result:
            stream_result = cast(StreamedRunResult[AgentContext[Any], Any], stream_result)
            original_stream = stream_result.stream

            async def wrapped_stream(*args, **kwargs):
                last_content = None
                async for chunk in original_stream(*args, **kwargs):
                    # Only emit if content has changed
                    if chunk != last_content:
                        self.chunk_streamed.emit(str(chunk), message_id)
                        last_content = chunk
                        yield chunk

                if stream_result.is_complete:
                    self.chunk_streamed.emit("", message_id)
                    messages = stream_result.new_messages()
                    if store_history:
                        self.conversation._last_messages = list(messages)
                        self.conversation.set_history(stream_result.all_messages())

                    # Extract and update tool calls
                    tool_calls = get_tool_calls(messages, dict(self.tool_manager._items))
                    for call in tool_calls:
                        call.message_id = message_id
                        call.context_data = self._context.data if self._context else None

                    # Format final content
                    responses = [m for m in messages if isinstance(m, ModelResponse)]
                    parts = [p for msg in responses for p in msg.parts]
                    content = "\n".join(format_part(p) for p in parts)
                    resolved_model = (
                        use_model.name()
                        if isinstance(use_model, Model)
                        else str(use_model)
                    )
                    # Update stream result with formatted content
                    stream_result.formatted_content = content  # type: ignore
                    stream_result.model_name = resolved_model  # type: ignore

            if model:
                original = self.model
                if isinstance(original, str):
                    original = infer_model(original)
                self.model_changed.emit(original)

            stream_result.stream = wrapped_stream  # type: ignore
            yield stream_result  # type: ignore
