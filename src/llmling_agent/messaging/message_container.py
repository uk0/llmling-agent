"""Message container with statistics and formatting capabilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from psygnal.containers import EventedList

from llmling_agent.log import get_logger
from llmling_agent.messaging.messages import ChatMessage


if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

    from llmling_agent.common_types import MessageRole
    from llmling_agent.messaging.messages import FormatStyle


logger = get_logger(__name__)


class ChatMessageContainer(EventedList[ChatMessage[Any]]):
    """Container for tracking and managing chat messages.

    Extends EventedList to provide:
    - Message statistics (tokens, costs)
    - History formatting
    - Token-aware context window management
    - Role-based filtering
    """

    def get_message_tokens(self, message: ChatMessage[Any]) -> int:
        """Get token count for a single message.

        Uses cost_info if available, falls back to tiktoken estimation.

        Args:
            message: Message to count tokens for

        Returns:
            Token count for the message
        """
        if message.cost_info:
            return message.cost_info.token_usage["total"]

        import tiktoken

        encoding = tiktoken.encoding_for_model(message.model or "gpt-3.5-turbo")
        content = "\n".join(message.format())
        return len(encoding.encode(content))

    def get_history_tokens(self) -> int:
        """Get total token count for all messages.

        Uses cost_info when available, falls back to tiktoken estimation
        for messages without usage information.

        Returns:
            Total token count across all messages
        """
        # Use cost_info if available
        total = sum(msg.cost_info.token_usage["total"] for msg in self if msg.cost_info)

        # For messages without cost_info, estimate using tiktoken
        if msgs := [msg for msg in self if not msg.cost_info]:
            import tiktoken

            model_name = next((m.model for m in self if m.model), "gpt-3.5-turbo")
            encoding = tiktoken.encoding_for_model(model_name)
            total += sum(len(encoding.encode(str(msg.content))) for msg in msgs)

        return total

    def get_total_cost(self) -> float:
        """Calculate total cost in USD across all messages.

        Only includes messages with cost information.

        Returns:
            Total cost in USD
        """
        return sum(float(msg.cost_info.total_cost) for msg in self if msg.cost_info)

    @property
    def last_message(self) -> ChatMessage[Any] | None:
        """Get most recent message or None if empty."""
        return self[-1] if self else None

    def format(
        self,
        *,
        style: FormatStyle = "simple",
        **kwargs: Any,
    ) -> str:
        """Format conversation history with configurable style.

        Args:
            style: Formatting style to use
            **kwargs: Additional formatting options passed to message.format()

        Returns:
            Formatted conversation history as string
        """
        return "\n".join(msg.format(style=style, **kwargs) for msg in self)

    def filter_by_role(
        self,
        role: MessageRole,
        *,
        max_messages: int | None = None,
    ) -> list[ChatMessage[Any]]:
        """Get messages with specific role.

        Args:
            role: Role to filter by (user/assistant/system)
            max_messages: Optional limit on number of messages to return

        Returns:
            List of messages with matching role
        """
        messages = [msg for msg in self if msg.role == role]
        if max_messages:
            messages = messages[-max_messages:]
        return messages

    def get_context_window(
        self,
        *,
        max_tokens: int | None = None,
        max_messages: int | None = None,
        include_system: bool = True,
    ) -> list[ChatMessage[Any]]:
        """Get messages respecting token and message limits.

        Args:
            max_tokens: Optional token limit for window
            max_messages: Optional message count limit
            include_system: Whether to include system messages

        Returns:
            List of messages fitting within constraints
        """
        # Filter system messages if needed
        history: Sequence[ChatMessage[Any]] = self
        if not include_system:
            history = [msg for msg in self if msg.role != "system"]

        # Apply message limit if specified
        if max_messages:
            history = history[-max_messages:]

        # Apply token limit if specified
        if max_tokens:
            token_count = 0
            filtered: list[Any] = []

            # Work backwards from most recent
            for msg in reversed(history):
                msg_tokens = self.get_message_tokens(msg)
                if token_count + msg_tokens > max_tokens:
                    break
                token_count += msg_tokens
                filtered.insert(0, msg)
            history = filtered

        return list(history)

    def get_between(
        self,
        *,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[ChatMessage[Any]]:
        """Get messages within a time range.

        Args:
            start_time: Optional start of range
            end_time: Optional end of range

        Returns:
            List of messages within the time range
        """
        messages = list(self)
        if start_time:
            messages = [msg for msg in messages if msg.timestamp >= start_time]
        if end_time:
            messages = [msg for msg in messages if msg.timestamp <= end_time]
        return messages
