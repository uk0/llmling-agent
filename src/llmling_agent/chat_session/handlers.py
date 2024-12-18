"""Event handlers for chat sessions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llmling_agent.chat_session.events import (
    SessionEvent,
    SessionEventHandler,
    SessionEventType,
)
from llmling_agent.log import get_logger
from llmling_agent.models.messages import ChatMessage


if TYPE_CHECKING:
    from llmling_agent.interfaces.ui import CoreUI


logger = get_logger(__name__)


class SessionUIHandler(SessionEventHandler):
    """Handles session events for UI interfaces."""

    def __init__(self, ui: CoreUI) -> None:
        """Initialize handler with UI interface.

        Args:
            ui: Any UI implementation that provides core UI capabilities
        """
        self.ui = ui

    async def handle_session_event(self, event: SessionEvent) -> None:
        """Handle a session event."""
        match event.type:
            case SessionEventType.HISTORY_CLEARED:
                await self.ui.send_message(
                    ChatMessage(content="History cleared", role="system")
                )
            case SessionEventType.SESSION_RESET:
                await self.ui.send_message(
                    ChatMessage(content="Session reset", role="system")
                )
                await self.ui.update_status("Session reset with default tools")
