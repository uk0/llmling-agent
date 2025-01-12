"""Storage provider interface for LLMling agent."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol


if TYPE_CHECKING:
    from datetime import datetime

    from llmling_agent.models.agents import ToolCallInfo
    from llmling_agent.models.messages import ChatMessage
    from llmling_agent.models.session import SessionQuery


class StorageProvider(Protocol):
    """Protocol defining the storage interface.

    Storage providers handle persistent storage of:
    - Messages and conversations
    - Tool calls and their results
    - Command history
    - Session data
    """

    async def initialize(self) -> None:
        """Initialize storage (create tables, connect to DB, etc.)."""
        ...

    async def cleanup(self) -> None:
        """Clean up resources."""
        ...

    async def filter_messages(
        self,
        query: SessionQuery,
    ) -> list[ChatMessage[str]]:
        """Get messages matching query criteria.

        Args:
            query: Filter criteria for messages

        Returns:
            List of messages matching the criteria
        """
        ...

    async def log_message(
        self,
        *,
        conversation_id: str,
        content: str,
        role: str,
        name: str | None = None,
        cost_info: Any | None = None,
        model: str | None = None,
        response_time: float | None = None,
        forwarded_from: list[str] | None = None,
    ) -> None:
        """Log a message.

        Args:
            conversation_id: ID of the conversation
            content: Message content
            role: Role of sender (user/assistant/system)
            name: Optional display name
            cost_info: Optional token/cost tracking
            model: Optional model identifier
            response_time: Optional response timing
            forwarded_from: Optional chain of forwarding agents
        """
        ...

    async def log_conversation(
        self,
        *,
        conversation_id: str,
        agent_name: str,
        start_time: datetime | None = None,
    ) -> None:
        """Log start of a conversation.

        Args:
            conversation_id: Unique conversation identifier
            agent_name: Name of the handling agent
            start_time: When conversation started
        """
        ...

    async def log_tool_call(
        self,
        *,
        conversation_id: str,
        message_id: str,
        tool_call: ToolCallInfo,
    ) -> None:
        """Log a tool call.

        Args:
            conversation_id: ID of conversation
            message_id: ID of message that triggered the call
            tool_call: Complete tool call information
        """
        ...

    async def log_command(
        self,
        *,
        agent_name: str,
        session_id: str,
        command: str,
    ) -> None:
        """Log a command execution.

        Args:
            agent_name: Name of agent that handled command
            session_id: Current session ID
            command: The command that was executed
        """
        ...

    async def get_commands(
        self,
        agent_name: str,
        session_id: str,
        *,
        limit: int | None = None,
        current_session_only: bool = False,
    ) -> list[str]:
        """Get command history.

        Args:
            agent_name: Agent to get history for
            session_id: Current session ID
            limit: Max number of commands to return
            current_session_only: Whether to only include current session

        Returns:
            List of command strings
        """
        ...

    @property
    def can_load_history(self) -> bool:
        """Whether this provider supports loading history."""
        ...
