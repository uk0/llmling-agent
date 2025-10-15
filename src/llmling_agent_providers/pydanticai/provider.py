"""Agent provider implementations."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import asdict
from decimal import Decimal
from functools import wraps
from typing import TYPE_CHECKING, Any, cast, get_args, get_origin

from llmling_models import AllModels, infer_model
import logfire
from pydantic_ai import Agent as PydanticAgent
import pydantic_ai._function_schema
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    TextPartDelta,
    ToolCallPart,
)
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.run import AgentRun
from pydantic_ai.tools import GenerateToolJsonSchema, RunContext
from pydantic_ai.usage import UsageLimits as PydanticAiUsageLimits

from llmling_agent.agent.context import AgentContext
from llmling_agent.common_types import ModelProtocol
from llmling_agent.log import get_logger
from llmling_agent.messaging.messages import TokenCost, TokenUsage
from llmling_agent.tasks.exceptions import (
    ChainAbortedError,
    RunAbortedError,
    ToolSkippedError,
)
from llmling_agent.utils.inspection import execute, has_argument_type
from llmling_agent_providers.base import AgentLLMProvider, ProviderResponse
from llmling_agent_providers.pydanticai.utils import (
    convert_prompts_to_user_content,
    get_tool_calls,
    to_model_request,
)


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

    from pydantic_ai import AgentStreamEvent
    from pydantic_ai.agent import AgentRunResult
    from pydantic_ai.run import AgentRunResultEvent

    from llmling_agent.common_types import EndStrategy, ModelType
    from llmling_agent.messaging.messages import ChatMessage
    from llmling_agent.models.content import Content
    from llmling_agent.tools.base import Tool
    from llmling_agent_providers.base import UsageLimits


logger = get_logger(__name__)


# 🤫 Secret agent stuff below
# We're doing some "creative" context handling here to make pydantic-ai accept our
# AgentContext.
# If you're reading this, please don't judge too harshly.
# Sometimes you gotta do what you gotta do to keep the architecture clean(ish).


def _is_call_ctx(annotation: Any) -> bool:
    # Yes, we're monkey-patching pydantic-ai's internal function.
    # No, we're not proud of it.
    # But it works™

    from pydantic._internal import _typing_extra

    if annotation is RunContext or (
        _typing_extra.is_generic_alias(annotation)
        and get_origin(annotation) is RunContext
    ):
        return True
    return annotation is AgentContext or (
        _typing_extra.is_generic_alias(annotation)
        and get_origin(annotation) is AgentContext
    )


pydantic_ai._function_schema._is_call_ctx = _is_call_ctx  # type: ignore


class PydanticAIProvider[TDeps](AgentLLMProvider[TDeps]):
    """Provider using pydantic-ai as backend."""

    NAME = "pydantic_ai"

    def __init__(
        self,
        *,
        model: str | ModelProtocol | None = None,
        context: AgentContext | None = None,
        name: str = "agent",
        retries: int = 1,
        output_retries: int | None = None,
        end_strategy: EndStrategy = "early",
        defer_model_check: bool = False,
        debug: bool = False,
        model_settings: dict[str, Any] | None = None,
    ):
        """Initialize pydantic-ai backend.

        Args:
            model: Model to use for responses
            context: Agent context
            name: Agent name
            retries: Number of retries for failed operations
            output_retries: Max retries for result validation
            end_strategy: How to handle tool calls with final result
            defer_model_check: Whether to defer model validation
            debug: Whether to enable debug mode
            model_settings: Additional model-specific settings
        """
        super().__init__(
            model=model,
            model_settings=model_settings,
            debug=debug,
            context=context,
        )
        self._kwargs: dict[str, Any] = dict(
            model=model,
            name=name,
            tools=[],
            retries=retries,
            end_strategy=end_strategy,
            output_retries=output_retries,
            defer_model_check=defer_model_check,
        )

    async def get_model_names(self) -> list[str]:
        """Get list of all known model names."""
        return list(get_args(KnownModelName)) + list(get_args(AllModels))

    async def get_agent(
        self,
        system_prompt: str,
        tools: list[Tool],
    ) -> PydanticAgent[Any, Any]:
        kwargs = self._kwargs.copy()
        model = kwargs.pop("model", None)
        model = infer_model(model) if isinstance(model, str) else model
        agent = PydanticAgent(model=model, instructions=system_prompt, **kwargs)  # type: ignore
        for tool in tools:
            wrapped = (
                self.wrap_tool(tool, self._context)
                if self._context
                else tool.callable.callable
            )
            if has_argument_type(wrapped, RunContext):
                agent.tool(wrapped)
            elif has_argument_type(wrapped, AgentContext):
                agent._function_toolset.add_function(
                    func=wrapped,
                    takes_ctx=True,
                    name=None,
                    retries=1,
                    prepare=None,
                    docstring_format="auto",
                    require_parameter_descriptions=False,
                    schema_generator=GenerateToolJsonSchema,
                    strict=None,
                )
            else:
                agent.tool_plain(wrapped)
        return agent

    @property
    def model(self) -> str | ModelType:
        return self._model

    def wrap_tool(
        self,
        tool: Tool,
        agent_ctx: AgentContext,
    ) -> Callable[..., Awaitable[Any]]:
        """Wrap tool with confirmation handling.

        We wrap the tool to intercept pydantic-ai's tool calls and add our confirmation
        logic before the actual execution happens. The actual tool execution (including
        moving sync functions to threads) is handled by pydantic-ai.

        Current situation is: We only get all infos for tool calls for functions with
        RunContext. In order to migitate this, we "fallback" to the AgentContext, which
        at least provides some information.
        """
        original_tool = tool.callable.callable
        if has_argument_type(original_tool, RunContext):

            async def wrapped(ctx: RunContext[AgentContext], *args, **kwargs):  # pyright: ignore
                result = await agent_ctx.handle_confirmation(tool, kwargs)
                if agent_ctx.report_progress:
                    await agent_ctx.report_progress(ctx.run_step, None)
                match result:
                    case "allow":
                        return await execute(original_tool, ctx, *args, **kwargs)
                    case "skip":
                        msg = f"Tool {tool.name} execution skipped"
                        raise ToolSkippedError(msg)
                    case "abort_run":
                        msg = "Run aborted by user"
                        raise RunAbortedError(msg)
                    case "abort_chain":
                        msg = "Agent chain aborted by user"
                        raise ChainAbortedError(msg)

        elif has_argument_type(original_tool, AgentContext):

            async def wrapped(ctx: AgentContext, *args, **kwargs):  # pyright: ignore
                result = await agent_ctx.handle_confirmation(tool, kwargs)
                match result:
                    case "allow":
                        return await execute(original_tool, agent_ctx, *args, **kwargs)
                    case "skip":
                        msg = f"Tool {tool.name} execution skipped"
                        raise ToolSkippedError(msg)
                    case "abort_run":
                        msg = "Run aborted by user"
                        raise RunAbortedError(msg)
                    case "abort_chain":
                        msg = "Agent chain aborted by user"
                        raise ChainAbortedError(msg)

        else:

            async def wrapped(*args, **kwargs):  # pyright: ignore
                result = await agent_ctx.handle_confirmation(tool, kwargs)
                match result:
                    case "allow":
                        return await execute(original_tool, *args, **kwargs)
                    case "skip":
                        msg = f"Tool {tool.name} execution skipped"
                        raise ToolSkippedError(msg)
                    case "abort_run":
                        msg = "Run aborted by user"
                        raise RunAbortedError(msg)
                    case "abort_chain":
                        msg = "Agent chain aborted by user"
                        raise ChainAbortedError(msg)

        wraps(original_tool)(wrapped)  # pyright: ignore
        wrapped.__doc__ = tool.description
        wrapped.__name__ = tool.name
        return wrapped

    @logfire.instrument("Pydantic-AI call. model: {model} result type {result_type}.")
    async def generate_response(
        self,
        *prompts: str | Content,
        message_id: str,
        message_history: list[ChatMessage],
        result_type: type[Any] | None = None,
        model: ModelType = None,
        tools: list[Tool] | None = None,
        system_prompt: str | None = None,
        usage_limits: UsageLimits | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Generate response using pydantic-ai."""
        agent = await self.get_agent(system_prompt or "", tools=tools or [])
        use_model = model or self.model
        if isinstance(use_model, str):
            use_model = infer_model(use_model)
            self.model_changed.emit(use_model)
        try:
            # Convert prompts to pydantic-ai format
            converted_prompts = await convert_prompts_to_user_content(prompts)

            # Run with complete history
            to_use = model or self.model
            to_use = infer_model(to_use) if isinstance(to_use, str) else to_use
            limits = asdict(usage_limits) if usage_limits else {}
            result: AgentRunResult = await agent.run(
                converted_prompts,  # Pass converted prompts
                deps=self._context,  # type: ignore
                message_history=[to_model_request(m) for m in message_history],
                model=to_use,  # type: ignore
                output_type=result_type or str,
                model_settings=self.model_settings,  # type: ignore
                usage_limits=PydanticAiUsageLimits(**limits),
            )

            # Extract tool calls and set message_id
            new_msgs = result.new_messages()
            tool_dict = {i.name: i for i in tools or []}
            tool_calls = get_tool_calls(new_msgs, tool_dict, agent_name=self.name)
            for call in tool_calls:
                call.message_id = message_id
                call.context_data = self._context.data if self._context else None
                self.tool_used.emit(call)

            # Get the actual model name from pydantic-ai response
            resolved_model = result.response.model_name or ""
            usage = result.usage()
            responses = [m for m in new_msgs if isinstance(m, ModelResponse)]
            try:
                cost_sum = sum(i.cost().total_price for i in responses)
            except (LookupError, AssertionError):
                logger.debug("Error calculating cost for %r", resolved_model)
                cost_sum = Decimal(0)
            token_usage = TokenUsage(
                total=usage.total_tokens,
                prompt=usage.input_tokens,
                completion=usage.output_tokens,
            )
            cost_info = TokenCost(token_usage=token_usage, total_cost=Decimal(cost_sum))
            return ProviderResponse(
                content=result.output,
                tool_calls=tool_calls,
                cost_and_usage=cost_info,
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
                return model.model_name
            case _:
                msg = f"Invalid model type: {model}"
                raise ValueError(msg)

    @asynccontextmanager
    async def iterate_run[TResult](
        self,
        *prompts: str | Content,  # Provider signature
        message_id: str,
        message_history: list[ChatMessage[Any]],  # Use Any generic
        tools: list[Tool] | None = None,
        result_type: type[TResult] | None = None,
        usage_limits: UsageLimits | None = None,
        model: ModelType = None,
        system_prompt: str | None = None,
        **kwargs: Any,  # Analogous kwargs
    ) -> AsyncIterator[AgentRun[TDeps, TResult]]:  # Use Any generic for now
        """Starts an iterable agent run using pydantic-ai's Agent.iter."""
        agent = await self.get_agent(system_prompt or "", tools=tools or [])
        use_model = model or self.model
        original_model_signal = None
        if model:
            original_model_signal = self.model
            if isinstance(use_model, str):
                use_model = infer_model(use_model)
            # Don't emit model_changed here, agent.iter takes model directly
            # self.model_changed.emit(use_model) # NO EMIT NEEDED HERE
        elif isinstance(use_model, str):
            use_model = infer_model(use_model)

        limits = asdict(usage_limits) if usage_limits else {}
        pydantic_limits = PydanticAiUsageLimits(**limits)
        resolved_model_name = (
            use_model.model_name if isinstance(use_model, Model) else str(use_model)
        )

        try:
            converted_prompts = await convert_prompts_to_user_content(prompts)
            model_messages = [to_model_request(m) for m in message_history]
            msg = "Starting PydanticAI agent iteration run_id=%s with model=%s"
            logger.debug(msg, message_id, resolved_model_name)
            async with agent.iter(
                converted_prompts,
                deps=self._context,
                message_history=model_messages,
                model=model or self.model,  # type: ignore
                output_type=result_type or str,
                model_settings=self.model_settings,  # type: ignore
                usage_limits=pydantic_limits,
                **kwargs,
            ) as agent_run:
                # 7. Post-processing (Minimal for iter, just yield)
                # Add model name to the run object if possible/needed?
                # setattr(agent_run, 'model_name', resolved_model_name)

                yield cast(AgentRun[Any, Any], agent_run)  # Yield the run object
        except Exception as e:
            # Catch potential errors from argument mismatches in agent.iter
            logger.exception("Error calling agent.iter")
            msg = f"Iteration setup error: {e}"
            raise RuntimeError(msg) from e
        finally:
            if original_model_signal and isinstance(original_model_signal, str):
                original_model_signal = infer_model(original_model_signal)
            # self.model_changed.emit(original_model_signal)

    def _create_tool_call_info(
        self, tool_part: ToolCallPart, tool_dict: dict, message_id: str
    ):
        """Create ToolCallInfo from tool call part."""
        from llmling_agent.tools.tool_call_info import ToolCallInfo

        return ToolCallInfo(
            tool_name=tool_part.tool_name,
            args=tool_part.args_as_dict(),
            result=None,  # Will be filled when tool execution completes
            agent_name=self.name,
            tool_call_id=tool_part.tool_call_id,
            message_id=message_id,
            context_data=self._context.data if self._context else None,
        )

    async def stream_events(
        self,
        *prompts: str | Content,
        message_history: list[ChatMessage],
        message_id: str,
        result_type: type[Any] | None = None,
        model: ModelType = None,
        tools: list[Tool] | None = None,
        system_prompt: str | None = None,
        usage_limits: UsageLimits | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[AgentStreamEvent | AgentRunResultEvent]:
        """Stream events directly from pydantic-ai without wrapper complexity."""
        agent = await self.get_agent(system_prompt or "", tools=tools or [])
        use_model = model or self.model
        limits = asdict(usage_limits) if usage_limits else {}
        if isinstance(use_model, str):
            use_model = infer_model(use_model)

        if model:
            self.model_changed.emit(use_model)

        # Convert prompts to pydantic-ai format
        converted_prompts = await convert_prompts_to_user_content(prompts)
        # Convert all messages to pydantic-ai format
        model_messages = [to_model_request(m) for m in message_history]

        tool_dict = {i.name: i for i in tools or []}

        # Stream events directly from pydantic-ai
        async for event in agent.run_stream_events(
            converted_prompts,
            deps=self._context,
            message_history=model_messages,
            model=model or self.model,  # type: ignore
            output_type=result_type or str,
            model_settings=self.model_settings,  # type: ignore
            usage_limits=PydanticAiUsageLimits(**limits),
        ):
            # Emit signals for external consumers
            match event:
                case (
                    PartStartEvent(part=ToolCallPart() as tool_part)
                    | FunctionToolCallEvent(part=tool_part)
                ):
                    call_info = self._create_tool_call_info(
                        tool_part, tool_dict, message_id
                    )
                    self.tool_used.emit(call_info)
                case PartDeltaEvent(delta=TextPartDelta(content_delta=delta)):
                    self.chunk_streamed.emit(delta, message_id)

            yield event

        # Stream is complete - emit empty chunk to signal completion
        self.chunk_streamed.emit("", message_id)

        # Reset model signal if needed
        if model:
            original = self.model
            if isinstance(original, str):
                original = infer_model(original)
            self.model_changed.emit(original)


if __name__ == "__main__":
    import asyncio

    from llmling_agent.tools import Tool

    async def main():
        provider = PydanticAIProvider[Any](model="openai:gpt-5-nano")

        def write_poem(text: str) -> bool:
            """The ultimate poem writing tool."""
            return True

        tool = Tool.from_callable(write_poem)
        print("Testing our provider's stream_events method...")
        chunks = []
        final_result = None
        async for event in provider.stream_events(
            "Write a long poem using the tool",
            message_history=[],
            message_id="test-123",
            tools=[tool],
        ):
            from pydantic_ai import AgentRunResultEvent, PartDeltaEvent, ToolCallPartDelta

            match event:
                case PartDeltaEvent(delta=TextPartDelta(content_delta=delta)):
                    chunks.append(delta)
                    print(f"Text chunk: '{delta}'")
                case PartDeltaEvent(delta=ToolCallPartDelta(content_delta=delta)):
                    chunks.append(delta)
                    print(f"Tool call chunk: '{delta}'")
                case AgentRunResultEvent(result=result):
                    final_result = result
                    print(f"Final result: {result}")

        print(f"\nTotal chunks collected: {len(chunks)}")
        print(f"Full content: {''.join(chunks)}")
        assert final_result is not None
        print(f"Final result available: {final_result is not None}")
        print(f"Model: {final_result.response.model_name}")

    asyncio.run(main())
