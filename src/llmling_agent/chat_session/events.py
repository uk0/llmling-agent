from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


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
