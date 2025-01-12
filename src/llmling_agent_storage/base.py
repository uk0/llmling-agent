from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from datetime import datetime

    from llmling_agent.models.agents import ToolCallInfo
    from llmling_agent.models.messages import ChatMessage, TokenCost
    from llmling_agent.models.session import SessionQuery


class StorageProvider:
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
    ) -> None:
        self.log_messages = log_messages
        self.log_conversations = log_conversations
        self.log_tool_calls = log_tool_calls
        self.log_commands = log_commands

    async def initialize(self) -> None:
        """Initialize storage if needed."""

    async def cleanup(self) -> None:
        """Clean up resources if needed."""

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
    ) -> None:
        """Log a message (if supported)."""

    async def log_conversation(
        self,
        *,
        conversation_id: str,
        agent_name: str,
        start_time: datetime | None = None,
    ) -> None:
        """Log a conversation (if supported)."""

    async def log_tool_call(
        self,
        *,
        conversation_id: str,
        message_id: str,
        tool_call: ToolCallInfo,
    ) -> None:
        """Log a tool call (if supported)."""

    async def log_command(
        self,
        *,
        agent_name: str,
        session_id: str,
        command: str,
    ) -> None:
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
