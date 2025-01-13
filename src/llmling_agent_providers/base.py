"""Agent provider implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from psygnal import Signal

from llmling_agent.log import get_logger
from llmling_agent.models.agents import ToolCallInfo


if TYPE_CHECKING:
    from collections.abc import Sequence
    from contextlib import AbstractAsyncContextManager

    from pydantic_ai.result import StreamedRunResult
    from tokonomics import Usage as TokonomicsUsage

    from llmling_agent.agent.conversation import ConversationManager
    from llmling_agent.common_types import ModelProtocol, ModelType
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
        model: str | ModelProtocol | None = None,
        name: str = "agent",
        debug: bool = False,
        system_prompt: str | Sequence[str] = (),
    ):
        self._name = name
        self._model = model
        self._agent: Any = None

        self.system_prompt = (
            system_prompt if isinstance(system_prompt, str) else "\n".join(system_prompt)
        )
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
        prompt: str,
        message_id: str,
        *,
        result_type: type[Any] | None = None,
        model: ModelType = None,
        store_history: bool = True,
        **kwargs: Any,
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
        store_history: bool = True,
        **kwargs: Any,
    ) -> AbstractAsyncContextManager[StreamedRunResult]:  # type: ignore[type-var]
        """Stream a response. Must be implemented by providers."""
        raise NotImplementedError
