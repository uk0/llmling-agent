"""Output writer implementations for chat sessions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from slashed import OutputWriter

from llmling_agent.log import get_logger
from llmling_agent.models.messages import ChatMessage


logger = get_logger(__name__)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


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
