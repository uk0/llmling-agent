from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
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
    data: dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


class SessionEventHandler(Protocol):
    """Protocol for session event handlers."""

    async def handle_session_event(self, event: SessionEvent) -> None:
        """Handle a session event."""
        ...
