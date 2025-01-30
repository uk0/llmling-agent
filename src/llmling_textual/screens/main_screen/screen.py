from __future__ import annotations

from typing import TYPE_CHECKING, Any

from textual.containers import Horizontal
from textual.screen import Screen
from textual.widgets import Header

from llmling_textual.screens.main_screen.agent_list import AgentEntry, AgentList
from llmling_textual.widgets.chat_view import ChatView


if TYPE_CHECKING:
    from textual.app import ComposeResult

    from llmling_agent.delegation.pool import AgentPool


class MainScreen(Screen):
    """Main application screen."""

    def __init__(self, pool: AgentPool, *args: Any, **kwargs: Any) -> None:
        self.pool = pool
        super().__init__(*args, **kwargs)

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            yield AgentList()
            yield ChatView()

    def on_mount(self) -> None:
        """Set up initial state."""
        # Update agent list
        agent_list = self.query_one(AgentList)
        agent_list.update_agents(self.pool)

        # Focus first agent if any
        if entries := self.query(AgentEntry):
            entries.first().focus()
