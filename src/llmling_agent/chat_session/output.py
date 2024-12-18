"""Output writer implementations for chat sessions."""

from __future__ import annotations

import sys
from typing import Protocol

from llmling_agent.models.messages import ChatMessage


class OutputWriter(Protocol):
    """Interface for command output."""

    async def print(self, message: str) -> None:
        """Write a message to output."""
        ...


class MessageCallback(Protocol):
    """Protocol for message callbacks."""

    async def __call__(self, message: ChatMessage) -> None: ...


class AsyncOutputWriter(OutputWriter):
    """Output writer that sends messages via async callback."""

    def __init__(
        self,
        message_callback: MessageCallback,
    ) -> None:
        """Initialize writer with callback.

        Args:
            message_callback: Async function to call with messages
        """
        self._callback = message_callback

    async def print(self, message: str) -> None:
        """Send message through callback.

        Args:
            message: Message content to send
        """
        chat_message = ChatMessage(content=message, role="system")
        await self._callback(chat_message)


class DefaultOutputWriter(OutputWriter):
    """Default output implementation using rich if available."""

    def __init__(self) -> None:
        """Initialize output writer."""
        try:
            from rich.console import Console

            self._console: Console | None = Console()
        except ImportError:
            self._console = None

    async def print(self, message: str) -> None:
        """Write message to output.

        Uses rich.Console if available, else regular print().
        """
        if self._console is not None:
            self._console.print(message)
        else:
            print(message, file=sys.stdout)
