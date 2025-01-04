"""Agent provider implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from llmling import ToolError
from psygnal import Signal
from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from llmling_agent.log import get_logger
from llmling_agent.models.agents import ToolCallInfo
from llmling_agent.models.context import AgentContext
from llmling_agent.pydantic_ai_utils import format_part, get_tool_calls
from llmling_agent.utils.inspection import has_argument_type


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from pydantic_ai.agent import EndStrategy, models
    from pydantic_ai.messages import ModelMessage
    from tokonomics import Usage as TokonomicsUsage

    from llmling_agent.tools.manager import ToolManager


logger = get_logger(__name__)


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

    async def generate_response(
        self,
        prompt: str,
        *,
        result_type: type[Any] | None = None,
        deps: Any | None = None,
        message_history: list[ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | None = None,
    ) -> ProviderResponse:
        """Generate a response for the given prompt.

        Args:
            prompt: Text prompt to respond to
            result_type: Optional type for structured responses
            deps: Optional dependency injection data
            message_history: Optional previous messages for context
            model: Optional model override

        Returns:
            Response message with optional structured content
        """
        ...

    def stream_response(
        self,
        prompt: str,
        *,
        result_type: type[Any] | None = None,
        deps: Any | None = None,
        message_history: list[ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | None = None,
    ) -> AsyncIterator[ProviderResponse]: ...


class PydanticAIProvider(AgentProvider):
    """Provider using pydantic-ai as backend."""

    def __init__(
        self,
        *,
        model: str | models.Model | None = None,
        system_prompt: str | Sequence[str] = (),
        tools: ToolManager,
        retries: int = 1,
        result_retries: int | None = None,
        end_strategy: EndStrategy = "early",
        defer_model_check: bool = False,
        context: AgentContext[Any] | None = None,
    ):
        """Initialize pydantic-ai backend.

        Args:
            model: Model to use for responses
            system_prompt: Initial system instructions
            tools: Available tools
            retries: Number of retries for failed operations
            result_retries: Max retries for result validation
            end_strategy: How to handle tool calls with final result
            defer_model_check: Whether to defer model validation
            context: Optional agent context
        """
        self._tool_manager = tools
        self._model = model
        self._agent = PydanticAgent(
            model=model,  # type: ignore
            system_prompt=system_prompt,
            tools=[],
            retries=retries,
            end_strategy=end_strategy,
            result_retries=result_retries,
            defer_model_check=defer_model_check,
            deps_type=AgentContext,
        )
        self._context = context

    async def generate_response(
        self,
        prompt: str,
        *,
        result_type: type[Any] | None = None,
        deps: Any | None = None,
        message_history: list[ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | None = None,
    ) -> ProviderResponse:
        """Generate response using pydantic-ai.

        Args:
            prompt: Text prompt to respond to
            result_type: Optional type for structured responses
            deps: Optional dependency injection data
            message_history: Optional previous messages for context
            model: Optional model override

        Returns:
            Response message with optional structured content
        """
        if deps is not None and self._context is not None:
            self._context.data = deps
        # Update available tools
        self._update_tools()
        try:
            # Run through pydantic-ai
            result = await self._agent.run(
                prompt,
                deps=self._context,  # type: ignore
                message_history=message_history,
                model=model,
            )

            # Extract tool calls
            tool_calls = get_tool_calls(
                result.new_messages(), dict(self._tool_manager._items)
            )
            # Return raw response data
            usage = result.usage()
            data = result.data
            return ProviderResponse(content=data, tool_calls=tool_calls, usage=usage)

        except Exception as e:
            logger.exception("Error generating response")
            msg = f"Response generation failed: {e}"
            raise ToolError(msg) from e

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

    async def stream_response(
        self,
        prompt: str,
        *,
        result_type: type[Any] | None = None,
        deps: Any | None = None,
        message_history: list[ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | None = None,
    ) -> AsyncIterator[ProviderResponse]:
        """Stream response using pydantic-ai."""
        if deps is not None and self._context is not None:
            self._context.data = deps

        # Update available tools
        self._update_tools()

        try:
            async with self._agent.run_stream(
                prompt,
                deps=self._context,  # type: ignore
                message_history=message_history,
                model=model,
            ) as stream_result:
                # Stream intermediate chunks
                async for response in stream_result.stream():
                    self.chunk_streamed.emit(str(response))
                    # Yield intermediate responses without tool calls/usage
                    yield ProviderResponse(content=str(response))

                # Once stream is complete, yield final state with all metadata
                messages = stream_result.new_messages()
                if messages:  # Get content from final messages
                    content = "\n".join(
                        format_part(part) for msg in messages for part in msg.parts
                    )
                    tool_calls = get_tool_calls(messages, dict(self._tool_manager))
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

    def __init__(self, name: str | None = None):
        """Initialize human provider."""
        self.name = name or "human"

    async def generate_response(
        self,
        prompt: str,
        *,
        result_type: type[Any] | None = None,
        deps: Any | None = None,
        message_history: list[ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | None = None,
    ) -> ProviderResponse:
        """Get response through human input.

        Args:
            prompt: Text prompt to respond to
            result_type: Optional type for structured responses
            deps: Optional dependency injection data
            message_history: Optional previous messages for context
            model: Not used for human provider
        """
        # Show context if available
        if message_history:
            print("\nContext:")
            for msg in message_history:
                if isinstance(msg, ModelRequest):
                    parts = [p for p in msg.parts if isinstance(p, UserPromptPart)]
                    for part in parts:
                        print(f"User: {part.content}")
                elif isinstance(msg, ModelResponse):
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
                msg = f"Invalid response format: {e}"
                raise ToolError(msg) from e

        return ProviderResponse(
            content=content,
            tool_calls=[],  # Human providers don't use tools for now
            usage=None,  # No token usage for human responses
        )

    async def stream_response(
        self,
        prompt: str,
        *,
        result_type: type[Any] | None = None,
        deps: Any | None = None,
        message_history: list[ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | None = None,
    ) -> AsyncIterator[ProviderResponse]:
        msg = "Streaming not supported for human provider"
        if False:  # to make it a generator
            yield ProviderResponse(content="")
        raise NotImplementedError(msg)
