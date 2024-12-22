"""Output writer implementations for chat sessions."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Protocol

from llmling_agent.log import get_logger
from llmling_agent.models.messages import ChatMessage


logger = get_logger(__name__)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


class OutputWriter(Protocol):
    """Interface for command output."""

    async def print(self, message: str):
        """Write a message to output."""
        ...


class CallbackOutputWriter(OutputWriter):
    """Output writer that sends messages via async callback."""

    def __init__(
        self,
        message_callback: Callable[[ChatMessage], Awaitable[None]],
    ):
        """Initialize writer with callback."""
        self._callback = message_callback

    async def print(self, message: str):
        """Send message through callback."""
        logger.debug("CallbackOutputWriter printing: %s", message)
        chat_message: ChatMessage[str] = ChatMessage(content=message, role="system")
        await self._callback(chat_message)


class DefaultOutputWriter(OutputWriter):
    """Default output implementation using rich if available."""

    def __init__(self):
        """Initialize output writer."""
        try:
            from rich.console import Console

            self._console: Console | None = Console()
        except ImportError:
            self._console = None

    async def print(self, message: str):
        """Write message to output.

        Uses rich.Console if available, else regular print().
        """
        if self._console is not None:
            self._console.print(message)
        else:
            print(message, file=sys.stdout)
