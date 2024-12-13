from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime  # noqa: TC003
from enum import Enum
from typing import Any, Protocol


class SessionEventType(Enum):
    """Types of session events."""

    HISTORY_CLEARED = "history_cleared"
    SESSION_RESET = "session_reset"


@dataclass
class SessionEvent:
    """Event data for session state changes."""

    type: SessionEventType
    timestamp: datetime
    data: dict[str, Any]


class SessionEventHandler(Protocol):
    """Protocol for session event handlers."""

    async def handle_session_event(self, event: SessionEvent) -> None:
        """Handle a session event."""
        ...
