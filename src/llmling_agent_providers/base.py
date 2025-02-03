"""Agent provider implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from psygnal import Signal

from llmling_agent.log import get_logger
from llmling_agent.models.agents import ToolCallInfo


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from contextlib import AbstractAsyncContextManager

    import tokonomics
    from tokonomics.pydanticai_cost import Usage

    from llmling_agent.agent.context import AgentContext
    from llmling_agent.agent.conversation import ConversationManager
    from llmling_agent.common_types import ModelProtocol, ModelType
    from llmling_agent.messaging.messages import ChatMessage, TokenCost
    from llmling_agent.models.content import Content
    from llmling_agent.tools.base import ToolInfo
    from llmling_agent.tools.manager import ToolManager


logger = get_logger(__name__)


@dataclass
class ProviderResponse:
    """Raw response data from provider."""

    content: Any
    tool_calls: list[ToolCallInfo] = field(default_factory=list)
    model_name: str = ""
    cost_and_usage: TokenCost | None = None
    provider_extra: dict[str, Any] | None = None


@runtime_checkable
class StreamingResponseProtocol[TResult](Protocol):
    """Protocol for streaming responses.

    This matches PydanticAI's StreamedRunResult interface to make transition easier,
    but lives in our core package to remove the dependency.
    """

    model_name: str | None
    is_complete: bool
    formatted_content: TResult = None  # type: ignore

    def stream(self) -> AsyncIterator[TResult]:
        """Stream individual chunks as they arrive."""
        ...

    def usage(self) -> Usage:
        """Get token usage statistics if available."""
        ...


class AgentProvider[TDeps]:
    """Base class for agent providers."""

    tool_used = Signal(ToolCallInfo)
    chunk_streamed = Signal(str, str)
    model_changed = Signal(object)  # Model | None
    NAME: str

    def __init__(
        self,
        *,
        model: str | ModelProtocol | None = None,
        name: str = "agent",
        debug: bool = False,
    ):
        super().__init__()
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
        message_history: list[ChatMessage],
        result_type: type[Any] | None = None,
        model: ModelType = None,
        tools: list[ToolInfo] | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Generate a response. Must be implemented by providers."""
        raise NotImplementedError

    def stream_response(
        self,
        *prompts: str | Content,
        message_id: str,
        message_history: list[ChatMessage],
        result_type: type[Any] | None = None,
        model: ModelType = None,
        tools: list[ToolInfo] | None = None,
        store_history: bool = True,
        **kwargs: Any,
    ) -> AbstractAsyncContextManager[StreamingResponseProtocol]:
        """Stream a response. Must be implemented by providers."""
        raise NotImplementedError

    async def supports_feature(self, capability: Literal["vision"]) -> bool:
        """Check if provider supports a specific capability."""
        import tokonomics

        match capability:
            case "vision":
                if not self.model_name:
                    return False
                caps = await tokonomics.get_model_capabilities(self.model_name)
                return (caps and caps.supports_vision) or False
            case _:
                return False

    async def get_token_limits(self) -> tokonomics.TokenLimits | None:
        """Get token limits for the current model."""
        import tokonomics

        if not self.model_name:
            return None

        try:
            return await tokonomics.get_model_limits(self.model_name)
        except ValueError:
            logger.debug("Could not get token limits for model: %s", self.model_name)
            return None


class AgentLLMProvider[TDeps](AgentProvider[TDeps]):
    """Provider using LLM backend."""
