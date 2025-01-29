"""Main Textual supervisor application."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import DataTable, Footer, Header, Static

from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from llmling_agent import AgentPool
    from llmling_agent.common_types import StrPath

logger = get_logger(__name__)


class AgentList(DataTable):
    """List of agents with their status."""

    DEFAULT_CSS = """
    AgentList {
        height: 100%;
        border: solid $primary;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self.add_columns("Name", "Status", "Model")
        self.cursor_type = "row"
        self.zebra_stripes = True

    def update_agents(self, pool: AgentPool) -> None:
        """Update agent list from pool."""
        self.clear()
        for name in sorted(pool.list_agents()):
            agent = pool.get_agent(name)
            status = "🔄 busy" if agent.is_busy() else "⏳ idle"
            self.add_row(
                name,
                status,
                agent.model_name or "default",
                key=name,
            )


class SupervisorApp(App):
    """Textual interface for managing agents."""

    CSS = """
    Screen {
        layout: grid;
        grid-size: 2;
        grid-columns: 30% 70%;
    }

    #sidebar {
        height: 100%;
    }

    #main {
        height: 100%;
    }
    """

    def __init__(self, config_path: StrPath, *, title: str | None = None) -> None:
        super().__init__()
        self.config_path = config_path
        if title:
            self.title = title
        self.pool: AgentPool | None = None
        self._agent_list: AgentList | None = None

    def compose(self) -> ComposeResult:
        """Create app layout."""
        yield Header()

        # Left sidebar with agent list
        with Vertical(id="sidebar"):
            yield AgentList()

        # Main content area (temporary)
        with Vertical(id="main"):
            yield Static("Main content will go here")

        yield Footer()

    async def on_mount(self) -> None:
        """Initialize pool and UI when app mounts."""
        from llmling_agent import AgentPool

        try:
            # Initialize pool
            logger.info("Initializing agent pool from %s", self.config_path)
            self.pool = await AgentPool(self.config_path).__aenter__()

            # Get widgets
            self._agent_list = self.query_one(AgentList)

            # Initial UI update
            self._refresh_agent_list()
            # Set up periodic refresh
            self.set_interval(1.0, self._refresh_agent_list)

        except Exception:
            logger.exception("Failed to initialize agent pool")
            self.exit(message="Failed to initialize agent pool")

    async def on_unmount(self) -> None:
        """Clean up when app unmounts."""
        if self.pool:
            try:
                await self.pool.__aexit__(None, None, None)
            except Exception:
                logger.exception("Error during pool cleanup")

    def _refresh_agent_list(self) -> None:
        """Update agent list display."""
        if self.pool and self._agent_list:
            self._agent_list.update_agents(self.pool)


if __name__ == "__main__":
    app = SupervisorApp(
        "src/llmling_agent/config_resources/agents.yml", title="Agent Supervisor"
    )
    app.run()
