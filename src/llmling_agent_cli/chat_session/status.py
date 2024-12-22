from __future__ import annotations

from typing import TYPE_CHECKING

from rich.panel import Panel
from rich.table import Table


if TYPE_CHECKING:
    from rich.console import Console

    from llmling_agent.ui.status import StatusBar


def render_status_bar(status_bar: StatusBar, console: Console):
    """Render status bar to console."""
    status = Table.grid(padding=1)

    # Add columns for visible fields
    for field in status_bar.fields:
        if field.visible:
            status.add_column(style=field.style, justify=field.align)

    # Add values in one row
    status.add_row(*(f"{f.label}: {f.value}" for f in status_bar.fields if f.visible))
    panel = Panel(status, style="dim", padding=(0, 1))
    console.print(panel)
