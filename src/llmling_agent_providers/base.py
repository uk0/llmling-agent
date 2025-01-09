"""Agent provider implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from psygnal import Signal

from llmling_agent.log import get_logger
from llmling_agent.models.agents import ToolCallInfo


if TYPE_CHECKING:
    from contextlib import AbstractAsyncContextManager

    from pydantic_ai.models import Model
    from pydantic_ai.result import StreamedRunResult
    from tokonomics import Usage as TokonomicsUsage

    from llmling_agent.agent.conversation import ConversationManager
    from llmling_agent.common_types import ModelType
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


class AgentProvider[TDeps]:
    """Base class for agent providers."""

    tool_used = Signal(ToolCallInfo)
    chunk_streamed = Signal(str, str)
    model_changed = Signal(object)  # Model | None

    def __init__(
        self,
        *,
        context: AgentContext[TDeps],
        tools: ToolManager,
        conversation: ConversationManager,
        model: str | Model | None = None,
        name: str = "agent",
    ):
        self._name = name
        self._model = model
        self._agent: Any = None
        self._tool_manager = tools
        self._context = context
        self._conversation = conversation
        self._debug = False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name})"

    def set_model(
        self,
        model: ModelType,
    ) -> None:
        """Default no-op implementation for setting model."""

    @property
    def model_name(self) -> str | None:
        """Get model name."""
        return None

    @property
    def name(self) -> str:
        """Get agent name."""
        return self._agent.name or "agent"

    @name.setter
    def name(self, value: str | None) -> None:
        """Set agent name."""
        self._agent.name = value

    async def generate_response(
        self,
        prompt: str,
        message_id: str,
        *,
        result_type: type[Any] | None = None,
        model: ModelType = None,
    ) -> ProviderResponse:
        """Generate a response. Must be implemented by providers."""
        raise NotImplementedError

    def stream_response(
        self,
        prompt: str,
        message_id: str,
        *,
        result_type: type[Any] | None = None,
        model: ModelType = None,
    ) -> AbstractAsyncContextManager[StreamedRunResult]:  # type: ignore[type-var]
        """Stream a response. Must be implemented by providers."""
        raise NotImplementedError

    @property
    def context(self) -> AgentContext[TDeps]:
        """Get provider context."""
        return self._context

    @context.setter
    def context(self, value: AgentContext[TDeps]):
        """Set provider context."""
        self._context = value
