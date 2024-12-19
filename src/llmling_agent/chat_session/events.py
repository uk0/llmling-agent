from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol


@dataclass(frozen=True)
class HistoryClearedEvent:
    """Emitted when chat history is cleared."""

    session_id: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass(frozen=True)
class SessionResetEvent:
    """Emitted when session is reset."""

    session_id: str
    previous_tools: dict[str, bool]
    new_tools: dict[str, bool]
    timestamp: datetime = field(default_factory=datetime.now)


class SessionEventHandler(Protocol):
    """Protocol for session event handlers."""

    async def handle_session_event(
        self, event: SessionResetEvent | HistoryClearedEvent
    ) -> None:
        """Handle a session event."""
