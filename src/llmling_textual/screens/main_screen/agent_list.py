from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer
from textual.message import Message
from textual.widgets import Static


if TYPE_CHECKING:
    from textual.app import ComposeResult

    from llmling_agent import AnyAgent
    from llmling_agent.delegation.pool import AgentPool


class AgentEntry(Static, can_focus=True):  # type: ignore
    """Individual agent entry."""

    DEFAULT_CSS = """
    AgentEntry {
        height: 3;
        padding: 0 1;
        background: $surface;
        border: tall $primary;
        margin: 0 1 1 1;
    }

    AgentEntry:hover {
        background: $accent;
    }

    AgentEntry:focus {
        background: $accent;
        border: tall $accent;
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

    class Clicked(Message):
        """Emitted when entry is clicked."""

        def __init__(self, agent: AnyAgent) -> None:
            self.agent = agent
            super().__init__()

    def __init__(self, agent: AnyAgent[Any, Any]):
        super().__init__("")
        self.agent = agent
        self.add_class("busy" if agent.is_busy() else "idle")

    def compose(self) -> ComposeResult:
        """Create entry layout."""
        with Horizontal():
            yield Static("●" if self.agent.is_busy() else "○", classes="status")
            yield Static(self.agent.name, classes="name")
            provider_name = getattr(
                self.agent.provider, "NAME", self.agent.provider.__class__.__name__
            )
            yield Static(f"({provider_name})", classes="provider")

    def on_click(self) -> None:
        """Handle click event."""
        self.post_message(self.Clicked(self.agent))


class AgentList(ScrollableContainer):
    """List of available agents."""

    BINDINGS: ClassVar = [
        Binding("up", "cursor_up", "Previous agent", show=False),
        Binding("down", "cursor_down", "Next agent", show=False),
        Binding("enter", "select_agent", "Select agent", show=False),
    ]

    DEFAULT_CSS = """
    AgentList {
        width: 100%;
        height: 100%;
        background: $surface-darken-1;
        padding: 0;
    }

    AgentList > .header {
        background: $panel;
        padding: 1;
        height: 3;
        border-bottom: solid $panel-darken-2;
    }
    """

    def __init__(
        self,
        *,
        id: str | None = None,  # noqa: A002
        classes: str | None = None,
    ) -> None:
        super().__init__(id=id, classes=classes)
        self._entries: dict[str, AgentEntry] = {}

    def compose(self) -> ComposeResult:
        """Create initial layout."""
        yield Static("Available Agents", classes="header")

    def update_agents(self, pool: AgentPool) -> None:
        """Update agent list from pool."""
        current_agents = set(pool.agents.keys())
        existing_agents = set(self._entries.keys())

        # Remove old entries
        for name in existing_agents - current_agents:
            if entry := self._entries.pop(name, None):
                entry.remove()

        # Add new entries
        for name in current_agents - existing_agents:
            agent = pool.agents[name]
            entry = AgentEntry(agent)
            self._entries[name] = entry
            self.mount(entry)

        # Update existing entries
        for name in current_agents & existing_agents:
            agent = pool.agents[name]
            entry = self._entries[name]
            entry.add_class("busy" if agent.is_busy() else "idle")

    def action_cursor_up(self) -> None:
        """Move focus to previous agent."""
        entries = list(self.query(AgentEntry))
        if not entries:
            return

        focused = self.screen.focused
        if isinstance(focused, AgentEntry):
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
        if isinstance(focused, AgentEntry):
            try:
                current_idx = entries.index(focused)
                new_idx = (current_idx + 1) % len(entries)
                entries[new_idx].focus()
            except ValueError:
                entries[0].focus()
        else:
            entries[0].focus()

    def action_select_agent(self) -> None:
        """Select focused agent."""
        if (focused := self.screen.focused) and isinstance(focused, AgentEntry):
            self.post_message(focused.Clicked(focused.agent))
