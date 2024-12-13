from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from rich.panel import Panel
from rich.table import Table


if TYPE_CHECKING:
    from rich.console import Console

    from .config import SessionState


@dataclass
class StatusInfo:
    """Current status information."""

    model: str
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    messages: int
    duration: str
    context_size: int | None = None


class StatusBar:
    """Status bar for interactive session."""

    def __init__(self, console: Console) -> None:
        """Initialize status bar."""
        self.console = console

    def render(self, state: SessionState) -> None:
        """Render status bar with current state."""
        duration = datetime.now() - state.start_time
        hours, remainder = divmod(int(duration.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)

        info = StatusInfo(
            model=state.current_model or "default",
            total_tokens=state.total_tokens,
            prompt_tokens=state.prompt_tokens,
            completion_tokens=state.completion_tokens,
            messages=state.message_count,
            duration=f"{hours:02d}:{minutes:02d}:{seconds:02d}",
        )

        status = Table.grid(padding=1)
        status.add_column(style="cyan", justify="left")
        status.add_column(style="green", justify="right")
        status.add_column(style="yellow", justify="right")
        status.add_column(style="blue", justify="right")

        token_info = (
            f"Tokens: {info.total_tokens:,} "
            f"(P: {info.prompt_tokens:,} "
            f"C: {info.completion_tokens:,})"
        )

        status.add_row(
            f"Model: {info.model}",
            token_info,
            f"Messages: {info.messages}",
            f"Time: {info.duration}",
        )

        self.console.print(
            Panel(
                status,
                style="bold",
                padding=(0, 1),
            )
        )
