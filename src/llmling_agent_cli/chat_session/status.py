"""Status bar rendering helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal

from rich.panel import Panel
from rich.table import Table


if TYPE_CHECKING:
    from rich.console import Console

    from llmling_agent_cli.chat_session.models import SessionState


Alignment = Literal["left", "center", "right"]


@dataclass
class BaseField:
    """Field in a status bar."""

    label: ClassVar[str]
    style: str = "dim"
    align: Alignment = "right"
    icon: str | None = None
    tooltip: str | None = None
    value: str = ""
    visible: bool = True


class ModelField(BaseField):
    """Model information."""

    label = "Model"
    align = "left"
    icon = "🤖"
    tooltip = "Current language model"


class TokensField(BaseField):
    """Token usage information."""

    label = "Tokens"
    icon = "🎯"
    tooltip = "Token usage"


class CostField(BaseField):
    """Cost information."""

    label = "Cost"
    icon = "💰"
    tooltip = "Total cost in USD"


class MessagesField(BaseField):
    """Message count information."""

    label = "Messages"
    icon = "💬"
    tooltip = "Message count"


class TimeField(BaseField):
    """Session duration information."""

    label = "Time"
    icon = "⏱️"
    tooltip = "Session duration"


class StatusBar:
    """UI-agnostic status bar."""

    def __init__(self):
        self.model = ModelField()
        self.tokens = TokensField()
        self.cost = CostField()
        self.messages = MessagesField()
        self.time = TimeField()

    def update(self, state: SessionState):
        """Update field values from state."""
        self.model.value = state.current_model or "default"
        self.tokens.value = f"{state.total_tokens:,}"
        self.cost.value = f"${state.total_cost:.3f}"
        self.messages.value = str(state.message_count)
        self.time.value = state.duration

    @property
    def fields(self) -> list[BaseField]:
        """Get all fields."""
        return [self.model, self.tokens, self.cost, self.messages, self.time]


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
