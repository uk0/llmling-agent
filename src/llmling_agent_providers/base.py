"""Agent provider implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from psygnal import Signal
import tokonomics
from toprompt import to_prompt

from llmling_agent.log import get_logger
from llmling_agent.models.agents import ToolCallInfo


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence
    from contextlib import AbstractAsyncContextManager

    from pydantic_ai.result import StreamedRunResult
    from tokonomics import Usage as TokonomicsUsage

    from llmling_agent.agent.conversation import ConversationManager
    from llmling_agent.common_types import ModelProtocol, ModelType
    from llmling_agent.models.content import Content
    from llmling_agent.models.context import AgentContext
    from llmling_agent.tools.manager import ToolManager


logger = get_logger(__name__)


@dataclass
class ProviderResponse:
    """Raw response data from provider."""

    content: Any
    tool_calls: list[ToolCallInfo] = field(default_factory=list)
    model_name: str = ""
    usage: TokonomicsUsage | None = None


@runtime_checkable
class StreamingResponse(Protocol):
    """Protocol for streaming responses.

    This matches PydanticAI's StreamedRunResult interface to make transition easier,
    but lives in our core package to remove the dependency.
    """

    model_name: str | None
    """Name of the model generating the response."""

    is_complete: bool
    """Whether the streaming is finished."""

    async def stream(self) -> AsyncIterator[str]:
        """Stream individual chunks as they arrive."""
        ...

    def usage(self) -> dict[str, int] | None:
        """Get token usage statistics if available."""
        ...


class AgentProvider[TDeps]:
    """Base class for agent providers."""

    tool_used = Signal(ToolCallInfo)
    chunk_streamed = Signal(str, str)
    model_changed = Signal(object)  # Model | None

    def __init__(
        self,
        *,
        model: str | ModelProtocol | None = None,
        name: str = "agent",
        debug: bool = False,
    ):
        self._name = name
        self._model = model
        self._tool_manager: ToolManager | None = None
        self._context: AgentContext[TDeps] | None = None
        self._conversation: ConversationManager | None = None
        self._debug = debug

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name})"

    def set_model(self, model: ModelType):
        """Default no-op implementation for setting model."""

    @property
    def tool_manager(self) -> ToolManager:
        """Get tool manager."""
        if self._tool_manager is None:
            msg = "Tool manager not set"
            raise RuntimeError(msg)
        return self._tool_manager

    @tool_manager.setter
    def tool_manager(self, value: ToolManager):
        """Set tool manager."""
        self._tool_manager = value

    @property
    def context(self) -> AgentContext[TDeps]:
        """Get context."""
        if self._context is None:
            msg = "Context not set"
            raise RuntimeError(msg)
        return self._context

    @context.setter
    def context(self, value: AgentContext[TDeps]):
        """Set context."""
        self._context = value

    @property
    def conversation(self) -> ConversationManager:
        """Get conversation manager."""
        if self._conversation is None:
            msg = "Conversation manager not set"
            raise RuntimeError(msg)
        return self._conversation

    @conversation.setter
    def conversation(self, value: ConversationManager):
        """Set conversation manager."""
        self._conversation = value

    @property
    def model_name(self) -> str | None:
        """Get model name."""
        return None

    @property
    def name(self) -> str:
        """Get agent name."""
        return self._name or "agent"

    @name.setter
    def name(self, value: str | None):
        """Set agent name."""
        self._name = value or "agent"

    async def generate_response(
        self,
        *prompts: str | Content,
        message_id: str,
        result_type: type[Any] | None = None,
        model: ModelType = None,
        store_history: bool = True,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Generate a response. Must be implemented by providers."""
        raise NotImplementedError

    def stream_response(
        self,
        *prompts: str | Content,
        message_id: str,
        result_type: type[Any] | None = None,
        model: ModelType = None,
        store_history: bool = True,
        **kwargs: Any,
    ) -> AbstractAsyncContextManager[StreamedRunResult]:  # type: ignore[type-var]
        """Stream a response. Must be implemented by providers."""
        raise NotImplementedError

    @staticmethod
    async def format_prompts(prompts: Sequence[str | Content]) -> str:
        """Format prompts for human readability using to_prompt."""
        parts = [await to_prompt(p) for p in prompts]
        return "\n\n".join(parts)

    async def supports_feature(self, capability: Literal["vision"]) -> bool:
        """Check if provider supports a specific capability."""
        match capability:
            case "vision":
                if not self.model_name:
                    return False
                caps = await tokonomics.get_model_capabilities(self.model_name)
                return (caps and caps.supports_vision) or False
            case _:
                return False
