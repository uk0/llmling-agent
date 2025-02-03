from __future__ import annotations

from typing import TYPE_CHECKING

from textual.screen import Screen
from textual.widgets import Footer, Header

from llmling_textual.screens.chat_screen.screen import ChatScreen
from llmling_textual.screens.main_screen.agent_list import AgentEntry, AgentList
from llmling_textual.widgets.message_stream import MessageStream


if TYPE_CHECKING:
    from textual.app import ComposeResult

    from llmling_agent.delegation.pool import AgentPool


class MainScreen(Screen):
    """Main application screen."""

    DEFAULT_CSS = """
    MainScreen {
        layout: grid;
        grid-size: 2;
        grid-columns: 1fr 4fr;
        padding: 1;
    }

    #agent-list {
        height: 100%;
    }

    #message-stream {
        height: 100%;
    }
    """

    def __init__(self, pool: AgentPool) -> None:
        super().__init__()
        self.pool = pool
        self.agent_list = AgentList(id="agent-list")
        self.message_stream = MessageStream(pool)

        # Connect pool events
        self.pool.nodes.events.added.connect(self._on_agent_change)
        self.pool.nodes.events.removed.connect(self._on_agent_change)

    def on_mount(self) -> None:
        """Initialize screen."""
        # Initial agent list population
        self.agent_list.update_agents(self.pool)

    def compose(self) -> ComposeResult:
        """Create main layout."""
        yield Header()
        yield self.agent_list
        yield self.message_stream
        yield Footer()

    def _on_agent_change(self, *_) -> None:
        """Handle agent added/removed."""
        if self.agent_list:  # Check if widget still exists
            self.agent_list.update_agents(self.pool)

    def cleanup(self) -> None:
        """Disconnect event handlers."""
        # Disconnect pool events
        self.pool._items.events.added.disconnect(self._on_agent_change)
        self.pool._items.events.removed.disconnect(self._on_agent_change)

        # Clean up stream
        self.message_stream.cleanup()

    async def on_agent_entry_clicked(self, event: AgentEntry.Clicked) -> None:
        """Show chat screen for clicked agent."""
        await self.app.push_screen(ChatScreen(event.agent))
