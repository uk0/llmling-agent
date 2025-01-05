"""Agent provider implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from llmling import ToolError
import logfire
from psygnal import Signal
from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models import infer_model

from llmling_agent.log import get_logger
from llmling_agent.models.agents import ToolCallInfo
from llmling_agent.models.context import AgentContext
from llmling_agent.pydantic_ai_utils import format_part, get_tool_calls, to_result_schema
from llmling_agent.responses.models import BaseResponseDefinition, ResponseDefinition
from llmling_agent.utils.inspection import has_argument_type


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from pydantic_ai.agent import EndStrategy, models
    from tokonomics import Usage as TokonomicsUsage

    from llmling_agent.agent.conversation import ConversationManager
    from llmling_agent.tools.manager import ToolManager


logger = get_logger(__name__)


def get_textual_streaming_app():
    from textual.app import App
    from textual.events import Key  # noqa: TC002
    from textual.widgets import Input

    class StreamingInputApp(App):
        def __init__(self, chunk_callback):
            super().__init__()
            self.chunk_callback = chunk_callback
            self.buffer = []
            self.done = False

        def compose(self):
            yield Input(id="input")

        def on_input_changed(self, event: Input.Changed):
            # New character was typed
            if len(event.value) > len(self.buffer):
                new_char = event.value[len(self.buffer) :]
                self.chunk_callback(new_char)
            self.buffer = list(event.value)

        def on_key(self, event: Key):
            if event.key == "enter":
                self.done = True
                self.result = "".join(self.buffer)
                self.exit()

    return StreamingInputApp


@dataclass
class ProviderResponse:
    """Raw response data from provider."""

    content: Any
    tool_calls: list[ToolCallInfo] = field(default_factory=list)
    usage: TokonomicsUsage | None = None


@runtime_checkable
class AgentProvider(Protocol):
    """Protocol for agent response generation."""

    tool_used = Signal(ToolCallInfo)
    chunk_streamed = Signal(str)
    model_changed = Signal(object)  # Model | None
    model: Any
    _agent = None

    async def generate_response(
        self,
        prompt: str,
        message_id: str,
        *,
        result_type: type[Any] | None = None,
    ) -> ProviderResponse:
        """Generate a response for the given prompt.

        Args:
            prompt: Text prompt to respond to
            message_id: ID to assign to the response and tool calls
            result_type: Optional type for structured responses

        Returns:
            Response message with optional structured content
        """
        ...

    def stream_response(
        self,
        prompt: str,
        *,
        result_type: type[Any] | None = None,
    ) -> AsyncIterator[ProviderResponse]: ...


class PydanticAIProvider(AgentProvider):
    """Provider using pydantic-ai as backend."""

    def __init__(
        self,
        *,
        model: str | models.Model | None = None,
        system_prompt: str | Sequence[str] = (),
        tools: ToolManager,
        conversation: ConversationManager,
        retries: int = 1,
        result_retries: int | None = None,
        end_strategy: EndStrategy = "early",
        defer_model_check: bool = False,
        debug: bool = False,
        context: AgentContext[Any] | None = None,
        **kwargs,
    ):
        """Initialize pydantic-ai backend.

        Args:
            model: Model to use for responses
            system_prompt: Initial system instructions
            tools: Available tools
            conversation: Conversation manager
            retries: Number of retries for failed operations
            result_retries: Max retries for result validation
            end_strategy: How to handle tool calls with final result
            defer_model_check: Whether to defer model validation
            context: Optional agent context
            debug: Whether to enable debug mode
            kwargs: Additional arguments for PydanticAI agent
        """
        self._tool_manager = tools
        self._model = model
        self._conversation = conversation
        self._debug = debug
        self._agent = PydanticAgent(
            model=model,  # type: ignore
            system_prompt=system_prompt,
            tools=[],
            retries=retries,
            end_strategy=end_strategy,
            result_retries=result_retries,
            defer_model_check=defer_model_check,
            deps_type=AgentContext,
            **kwargs,
        )
        self._context = context

    @property
    def model(self) -> str | models.Model | models.KnownModelName | None:
        return self._model

    @model.setter
    def model(self, value: models.Model | models.KnownModelName | None):
        self._model = value
        self._agent.model = value

    @logfire.instrument("Pydantic-AI call. result type {result_type}. Prompt: {prompt}")
    async def generate_response(
        self,
        prompt: str,
        message_id: str,
        *,
        result_type: type[Any] | None = None,
        model: models.Model | models.KnownModelName | None = None,
    ) -> ProviderResponse:
        """Generate response using pydantic-ai.

        Args:
            prompt: Text prompt to respond to
            message_id: ID to assign to the response and tool calls
            result_type: Optional type for structured responses
            model: Optional model override for this call

        Returns:
            Response message with optional structured content
        """
        self._update_tools()
        message_history = self._conversation.get_history()

        # If we have a model override, temporarily set it
        original_model = self._agent.model
        try:
            result = await self._agent.run(
                prompt,
                deps=self._context,
                message_history=message_history,
                model=model or self.model,
            )

            # Extract tool calls and set message_id
            new_msgs = result.new_messages()
            tool_calls = get_tool_calls(new_msgs, self._tool_manager)
            for call in tool_calls:
                call.message_id = message_id
                call.context_data = self._context.data if self._context else None

            self._conversation._last_messages = list(new_msgs)
            self._conversation.set_history(result.all_messages())

            return ProviderResponse(
                content=result.data, tool_calls=tool_calls, usage=result.usage()
            )
        finally:
            # Restore original model if we had an override
            if model:
                self.model = original_model

    def _update_tools(self):
        """Update pydantic-ai-agent tools."""
        self._agent._function_tools.clear()
        tools = [t for t in self._tool_manager.values() if t.enabled]
        for tool in tools:
            wrapped = (
                self._context.wrap_tool(tool, self._context)
                if self._context
                else tool.callable.callable
            )
            if has_argument_type(wrapped, "RunContext"):
                self._agent.tool(wrapped)
            else:
                self._agent.tool_plain(wrapped)

    def set_model(self, model: models.Model | models.KnownModelName | None):
        """Set the model for this agent.

        Args:
            model: New model to use (name or instance)

        Emits:
            model_changed signal with the new model
        """
        old_name = self.model_name
        if isinstance(model, str):
            model = infer_model(model)
        self._agent.model = model
        self.model_changed.emit(model)
        logger.debug("Changed model from %s to %s", old_name, self.model_name)

    @property
    def model_name(self) -> str | None:
        """Get the model name in a consistent format."""
        match self._agent.model:
            case str() | None:
                return self._agent.model
            case _:
                return self._agent.model.name()

    def result_validator(self, *args: Any, **kwargs: Any) -> Any:
        """Register a result validator.

        Validators can access runtime through RunContext[AgentContext].

        Example:
            ```python
            @agent.result_validator
            async def validate(ctx: RunContext[AgentContext], result: str) -> str:
                if len(result) < 10:
                    raise ModelRetry("Response too short")
                return result
            ```
        """
        return self._agent.result_validator(*args, **kwargs)

    def set_result_type(
        self,
        result_type: type | str | ResponseDefinition | None,
        *,
        tool_name: str | None = None,
        tool_description: str | None = None,
    ):
        """Set or update the result type for this agent."""
        schema = to_result_schema(
            result_type,
            context=self._context,
            tool_name_override=tool_name,
            tool_description_override=tool_description,
        )
        logger.debug("Created schema: %s", schema)

        # Apply schema and settings
        self._agent._result_schema = schema
        self._agent._allow_text_result = schema.allow_text_result if schema else True

        # Apply retries if from response definition
        match result_type:
            case BaseResponseDefinition() if result_type.result_retries is not None:
                self._agent._max_result_retries = result_type.result_retries

    async def stream_response(
        self,
        prompt: str,
        *,
        result_type: type[Any] | None = None,
        model: models.Model | models.KnownModelName | None = None,
    ) -> AsyncIterator[ProviderResponse]:
        """Stream response using pydantic-ai.

        Args:
            prompt: Text prompt to respond to
            result_type: Optional type for structured responses
            model: Optional model override for this call

        Yields:
            Intermediate responses during streaming and final state

        Raises:
            ToolError: If response streaming fails
        """
        self._update_tools()

        try:
            message_history = self._conversation.get_history()

            if self._debug:
                from devtools import debug

                debug(self._agent)

            async with self._agent.run_stream(
                prompt,
                deps=self._context,  # type: ignore
                message_history=message_history,
                model=model or self.model,  # Use override if provided
            ) as stream_result:
                # Stream intermediate chunks
                async for response in stream_result.stream():
                    self.chunk_streamed.emit(str(response))
                    yield ProviderResponse(content=str(response))

                # Once stream is complete, update conversation and yield final state
                if stream_result.is_complete:
                    messages = stream_result.new_messages()
                    self._conversation._last_messages = list(messages)
                    self._conversation.set_history(stream_result.all_messages())

                    # Format final content and extract tool calls
                    content = "\n".join(format_part(p) for m in messages for p in m.parts)
                    tool_calls = get_tool_calls(messages, self._tool_manager)

                    yield ProviderResponse(
                        content=content,
                        tool_calls=tool_calls,
                        usage=stream_result.usage(),
                    )

        except Exception as e:
            logger.exception("Error streaming response")
            msg = f"Response streaming failed: {e}"
            raise ToolError(msg) from e


class HumanProvider(AgentProvider):
    """Provider for human-in-the-loop responses."""

    model = None

    def __init__(
        self,
        conversation: ConversationManager,
        name: str = "human",
        debug: bool = False,
    ):
        """Initialize human provider."""
        self.name = name or "human"
        self._conversation = conversation
        self._debug = debug

    @logfire.instrument("Human input. result type {result_type}. Prompt: {prompt}")
    async def generate_response(
        self,
        prompt: str,
        *,
        result_type: type[Any] | None = None,
    ) -> ProviderResponse:
        """Get response through human input.

        Args:
            prompt: Text prompt to respond to
            result_type: Optional type for structured responses
        """
        # Show context if available
        message_history = self._conversation.get_history()
        if message_history:
            print("\nContext:")
            for msg in message_history:
                match msg:
                    case ModelRequest():
                        parts = [p for p in msg.parts if isinstance(p, UserPromptPart)]
                        for part in parts:
                            print(f"User: {part.content}")
                    case ModelResponse():
                        parts = [p for p in msg.parts if isinstance(p, TextPart)]
                        for part in parts:
                            print(f"Assistant: {part.content}")
            print("\n---")

        # Show prompt and get response
        print(f"\n{prompt}")
        if result_type:
            print(f"(Please provide response as {result_type.__name__})")
        response = input("> ")

        # Parse structured response if needed
        content: Any = response
        if result_type:
            try:
                content = result_type.model_validate_json(response)
            except Exception as e:
                logger.exception("Failed to parse structured response")
                error_msg = f"Invalid response format: {e}"
                raise ToolError(error_msg) from e

        return ProviderResponse(content=content, tool_calls=[], usage=None)

    async def stream_response(
        self,
        prompt: str,
        *,
        result_type: type[Any] | None = None,
    ) -> AsyncIterator[ProviderResponse]:
        """Stream response keystroke by keystroke."""
        print(f"\n{prompt}")
        if result_type:
            print(f"(Please provide response as {result_type.__name__})")

        content_buffer = []

        async def handle_chunk(chunk: str):
            self.chunk_streamed.emit(chunk)
            content_buffer.append(chunk)
            yield ProviderResponse(content=chunk)

        textual_app = get_textual_streaming_app()
        app = textual_app(handle_chunk)
        await app.run_async()

        content = "".join(content_buffer)

        # Parse structured response if needed
        if result_type:
            try:
                content = result_type.model_validate_json(content)
            except Exception as e:
                logger.exception("Failed to parse structured response")
                error_msg = f"Invalid response format: {e}"
                raise ToolError(error_msg) from e

        # Yield final response
        yield ProviderResponse(content=content)
