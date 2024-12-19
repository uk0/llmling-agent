"""Status bar for interactive session."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Literal, Protocol

from rich.panel import Panel
from rich.table import Table

from llmling_agent.chat_session.models import SessionState


if TYPE_CHECKING:
    from collections.abc import Callable

    from rich.console import Console


Alignment = Literal["left", "center", "right"]


class StatusBarField(Protocol):
    """A field in the status bar."""

    @property
    def label(self) -> str:
        """Get field label."""
        ...

    @property
    def style(self) -> str:
        """Get field style."""
        ...

    @property
    def align(self) -> Alignment:
        """Get field alignment."""
        ...

    def should_show(self, info: SessionState) -> bool:
        """Determine if field should be shown."""
        ...

    def get_value(self, info: SessionState) -> str:
        """Get formatted value for this field."""
        ...


@dataclass
class BaseField:
    """Base implementation of a status bar field."""

    label: str
    style: str = "dim"
    align: Alignment = "right"
    condition: Callable[[SessionState], bool] | None = None

    def should_show(self, info: SessionState) -> bool:
        """Check if field should be shown."""
        if self.condition is None:
            return True
        return self.condition(info)


@dataclass
class ModelField(BaseField):
    """Shows current model."""

    def __init__(
        self,
        style: str = "dim",
        align: Alignment = "left",
        condition: Callable[[SessionState], bool] | None = None,
    ):
        super().__init__("Model", style, align, condition)

    def get_value(self, info: SessionState) -> str:
        return info.current_model or "default"


@dataclass
class TokensField(BaseField):
    """Shows token usage."""

    def __init__(
        self,
        style: str = "dim",
        align: Alignment = "right",
        condition: Callable[[SessionState], bool] | None = None,
    ):
        super().__init__("Tokens", style, align, condition)

    def get_value(self, info: SessionState) -> str:
        return (
            f"{info.total_tokens:,} "
            f"(Prompt: {info.prompt_tokens:,} "
            f"Completion: {info.completion_tokens:,})"
        )


@dataclass
class CostField(BaseField):
    """Shows total cost."""

    def __init__(
        self,
        style: str = "dim",
        align: Alignment = "right",
        condition: Callable[[SessionState], bool] | None = lambda i: i.total_cost > 0,
    ):
        super().__init__("Cost", style, align, condition)

    def get_value(self, info: SessionState) -> str:
        return f"${info.total_cost:.3f}"


@dataclass
class MessagesField(BaseField):
    """Shows message count."""

    def __init__(
        self,
        style: str = "dim",
        align: Alignment = "right",
        condition: Callable[[SessionState], bool] | None = None,
    ):
        super().__init__("Messages", style, align, condition)

    def get_value(self, info: SessionState) -> str:
        return str(info.message_count)


@dataclass
class TimeField(BaseField):
    """Shows session duration."""

    def __init__(
        self,
        style: str = "dim",
        align: Alignment = "right",
        condition: Callable[[SessionState], bool] | None = None,
    ):
        super().__init__("Time", style, align, condition)

    def get_value(self, info: SessionState) -> str:
        return info.duration


class StatusBar:
    """Status bar for interactive session."""

    def __init__(
        self,
        console: Console,
        fields: list[StatusBarField] | None = None,
    ):
        """Initialize status bar."""
        self.console = console
        # Default fields if none provided
        self.fields = fields or [
            ModelField(),
            TokensField(),
            CostField(),
            MessagesField(),
            TimeField(),
        ]

    def render(self, state: SessionState):
        """Render status bar with current state."""
        duration = datetime.now() - state.start_time
        hours, remainder = divmod(int(duration.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)

        info = SessionState(
            current_model=state.current_model or "default",
            total_tokens=state.total_tokens,
            prompt_tokens=state.prompt_tokens,
            completion_tokens=state.completion_tokens,
            total_cost=state.total_cost,
            message_count=state.message_count,
            start_time=state.start_time,
        )

        status = Table.grid(padding=1)
        visible_fields = [f for f in self.fields if f.should_show(info)]

        # Add a column for each visible field
        for field in visible_fields:
            status.add_column(style=field.style, justify=field.align)
        # Add all fields in one row
        status.add_row(*(f"{f.label}: {f.get_value(info)}" for f in visible_fields))
        panel = Panel(status, style="dim", padding=(0, 1), border_style="dim")
        self.console.print(panel)
