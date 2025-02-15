"""Main application."""

from __future__ import annotations

import logging
from typing import ClassVar

from textual.app import App
from textual.binding import Binding
from textualicious import LoggingWidget

from llmling_agent.delegation.pool import AgentPool
from llmling_agent_input.textual_provider import TextualInputProvider
from llmling_textual.screens.log_screen import LogWindow
from llmling_textual.screens.main_screen import MainScreen
from llmling_textual.screens.pool_selection_screen import PoolSelectionScreen


class PoolApp(App):
    """Main application."""

    DEFAULT_CSS = """
    #main-container {
        height: 100%;
        width: 100%;
    }
    """
    BINDINGS: ClassVar = [
        Binding("f12", "toggle_logs", "Show/Hide Logs"),
        Binding("p", "select_pool", "Select Pool"),
    ]

    def __init__(self, pool: AgentPool):
        super().__init__()
        self.agent_pool = pool
        self.main_screen: MainScreen | None = None
        self.log_widget = LoggingWidget()
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        logger.addHandler(self.log_widget.handler)

        # Set up input provider for the pool
        self.agent_pool._input_provider = TextualInputProvider(self)

    async def on_mount(self):
        """Initialize pool and mount main screen."""
        await self.agent_pool.__aenter__()

        self.main_screen = MainScreen(self.agent_pool)
        await self.push_screen(self.main_screen)

    async def on_unmount(self):
        """Clean up resources."""
        if self.main_screen:
            # First disconnect event handlers
            self.main_screen.cleanup()

        await self.agent_pool.__aexit__(None, None, None)

    def action_toggle_logs(self):
        """Toggle log window."""
        if isinstance(self.screen, LogWindow):
            self.pop_screen()
        else:
            self.push_screen(LogWindow(self.log_widget))

    def action_select_pool(self):
        """Show pool selection screen."""
        self.push_screen(PoolSelectionScreen())


if __name__ == "__main__":
    # Example of direct usage (for development/testing)
    from llmling_agent import AgentPool

    path = "src/llmling_agent/config_resources/agents_template.yml"
    pool = AgentPool[None](path)
    app = PoolApp(pool)
    app.run()
