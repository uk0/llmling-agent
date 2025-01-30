from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer
from textual.widgets import Static

from llmling_textual.screens.chat_screen.screen import ChatScreen


if TYPE_CHECKING:
    from textual.app import ComposeResult

    from llmling_agent import AnyAgent


class AgentEntry(Static, can_focus=True):  # type: ignore
    """Individual agent entry."""

    DEFAULT_CSS = """
    AgentEntry {
        height: 1;
        padding: 0 1;
        background: $surface;
    }

    AgentEntry:hover {
        background: $accent;
    }

    AgentEntry:focus {
        background: $accent;
        border: none;
    }

    AgentEntry.busy {
        color: $warning;
    }

    AgentEntry > Horizontal {
        height: 1;
        width: 100%;
    }

    AgentEntry .status {
        width: 2;
        color: $text-muted;
    }

    AgentEntry .name {
        width: auto;
        margin-right: 1;
    }

    AgentEntry .provider {
        color: $text-muted;
    }
    """

    def __init__(self, agent: AnyAgent[Any, Any]):
        super().__init__()
        self.agent = agent
        self.add_class("busy" if agent.is_busy() else "idle")

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Static("●" if self.agent.is_busy() else "○", classes="status")
            yield Static(self.agent.name, classes="name")
            # Get provider name, fallback to class name if not available
            provider_name = self.agent.provider.NAME
            yield Static(f"({provider_name})", classes="provider")


class AgentList(ScrollableContainer):
    """List of available agents."""

    BINDINGS: ClassVar = [
        Binding("up", "cursor_up", "Previous agent", show=False),
        Binding("down", "cursor_down", "Next agent", show=False),
        Binding("enter", "select_agent", "Select agent", show=False),
    ]

    DEFAULT_CSS = """
    AgentList {
        width: 30%;
        background: $surface-darken-1;
        padding: 0;
    }

    AgentList > .header {
        background: $panel;
        padding: 1;
        border-bottom: solid $panel-darken-2;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("Available Agents", classes="header")

    def update_agents(self, pool) -> None:
        """Update agent list from pool."""
        # Remove old entries after header
        for child in self.query(AgentEntry):
            child.remove()

        # Add new entries
        for agent in pool.agents.values():
            self.mount(AgentEntry(agent))

        # Focus first entry if any
        if entries := self.query(AgentEntry):
            entries.first().focus()

    def action_cursor_up(self) -> None:
        """Move focus to previous agent."""
        entries = list(self.query(AgentEntry))
        if not entries:
            return

        focused = self.screen.focused
        if isinstance(focused, AgentEntry):  # Type check
            try:
                current_idx = entries.index(focused)
                new_idx = (current_idx - 1) % len(entries)
                entries[new_idx].focus()
            except ValueError:
                entries[-1].focus()
        else:
            entries[0].focus()

    def action_cursor_down(self) -> None:
        """Move focus to next agent."""
        entries = list(self.query(AgentEntry))
        if not entries:
            return

        focused = self.screen.focused
        if isinstance(focused, AgentEntry):  # Type check
            try:
                current_idx = entries.index(focused)
                new_idx = (current_idx + 1) % len(entries)
                entries[new_idx].focus()
            except ValueError:
                entries[0].focus()
        else:
            entries[0].focus()

    def action_select_agent(self) -> None:
        """Handle agent selection."""
        if (focused := self.screen.focused) and isinstance(focused, AgentEntry):
            self.app.push_screen(ChatScreen(focused.agent))
