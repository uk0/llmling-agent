from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

from llmling_agent.utils.tasks import TaskManagerMixin


if TYPE_CHECKING:
    from datetime import datetime

    from llmling_agent.models.agents import ToolCallInfo
    from llmling_agent.models.messages import ChatMessage, TokenCost
    from llmling_agent.models.session import SessionQuery

T = TypeVar("T")


class StorageProvider(TaskManagerMixin):
    """Base class for storage providers."""

    can_load_history: bool = False
    """Whether this provider supports loading history."""

    def __init__(
        self,
        *,
        log_messages: bool = True,
        log_conversations: bool = True,
        log_tool_calls: bool = True,
        log_commands: bool = True,
        log_context: bool = True,
    ):
        super().__init__()
        self.log_messages = log_messages
        self.log_conversations = log_conversations
        self.log_tool_calls = log_tool_calls
        self.log_commands = log_commands
        self.log_context = log_context

    def cleanup(self):
        """Clean up resources."""

    async def filter_messages(
        self,
        query: SessionQuery,
    ) -> list[ChatMessage[str]]:
        """Get messages matching query (if supported)."""
        msg = f"{self.__class__.__name__} does not support loading history"
        raise NotImplementedError(msg)

    async def log_message(
        self,
        *,
        conversation_id: str,
        content: str,
        role: str,
        name: str | None = None,
        cost_info: TokenCost | None = None,
        model: str | None = None,
        response_time: float | None = None,
        forwarded_from: list[str] | None = None,
    ):
        """Log a message (if supported)."""

    async def log_conversation(
        self,
        *,
        conversation_id: str,
        agent_name: str,
        start_time: datetime | None = None,
    ):
        """Log a conversation (if supported)."""

    async def log_tool_call(
        self,
        *,
        conversation_id: str,
        message_id: str,
        tool_call: ToolCallInfo,
    ):
        """Log a tool call (if supported)."""

    async def log_command(self, *, agent_name: str, session_id: str, command: str):
        """Log a command (if supported)."""

    async def get_commands(
        self,
        agent_name: str,
        session_id: str,
        *,
        limit: int | None = None,
        current_session_only: bool = False,
    ) -> list[str]:
        """Get command history (if supported)."""
        msg = f"{self.__class__.__name__} does not support retrieving commands"
        raise NotImplementedError(msg)

    async def log_context_message(
        self,
        *,
        conversation_id: str,
        content: str,
        role: str,
        name: str | None = None,
        model: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """Log a context message if context logging is enabled."""
        if not self.log_context:
            return

        await self.log_message(
            conversation_id=conversation_id,
            content=content,
            role=role,
            name=name,
            model=model,
        )

    # Sync wrapper
    def log_context_message_sync(self, **kwargs):
        """Sync wrapper for log_context_message."""
        self.fire_and_forget(self.log_context_message(**kwargs))

    # Sync wrappers for all async methods
    def log_message_sync(self, **kwargs):
        """Sync wrapper for log_message."""
        self.fire_and_forget(self.log_message(**kwargs))

    def log_conversation_sync(self, **kwargs):
        """Sync wrapper for log_conversation."""
        self.fire_and_forget(self.log_conversation(**kwargs))

    def log_tool_call_sync(self, **kwargs):
        """Sync wrapper for log_tool_call."""
        self.fire_and_forget(self.log_tool_call(**kwargs))

    def log_command_sync(self, **kwargs):
        """Sync wrapper for log_command."""
        self.fire_and_forget(self.log_command(**kwargs))

    def get_commands_sync(self, **kwargs) -> list[str]:
        """Sync wrapper for get_commands."""
        return self.run_sync(self.get_commands(**kwargs))

    def filter_messages_sync(self, **kwargs) -> list[ChatMessage[str]]:
        """Sync wrapper for filter_messages."""
        return self.run_sync(self.filter_messages(**kwargs))
