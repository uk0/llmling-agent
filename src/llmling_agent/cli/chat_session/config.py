from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import os
from pathlib import Path
import platform


@dataclass
class SessionState:
    """Current state of the interactive session."""

    current_model: str | None = None
    message_count: int = 0
    total_tokens: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    last_command: str | None = None


def get_history_file(agent_name: str) -> Path:
    """Get history file path for agent."""
    if platform.system() == "Windows":
        base = Path(os.getenv("APPDATA", "")) / "llmling"
    else:
        base = Path.home() / ".local" / "share" / "llmling"

    history_dir = base / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    return history_dir / f"{agent_name}.history"
