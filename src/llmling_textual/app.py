from __future__ import annotations

from textual.app import App

from llmling_agent.delegation.pool import AgentPool
from llmling_textual.screens.main_screen import MainScreen


class AgentApp(App):
    """Main application."""

    DEFAULT_CSS = """
    #main-container {
        height: 100%;
        width: 100%;
    }
    """

    def __init__(self, config_path: str):
        super().__init__()
        self._config_path = config_path
        self.agent_pool: AgentPool | None = None
        self.main_screen: MainScreen | None = None

    async def on_mount(self) -> None:
        """Initialize pool and mount main screen."""
        self.agent_pool = AgentPool(self._config_path)
        await self.agent_pool.__aenter__()

        self.main_screen = MainScreen(self.agent_pool)
        await self.push_screen(self.main_screen)

    async def on_unmount(self) -> None:
        """Clean up resources."""
        if self.main_screen:
            # First disconnect event handlers
            self.main_screen.cleanup()

        if self.agent_pool:
            await self.agent_pool.__aexit__(None, None, None)


if __name__ == "__main__":
    app = AgentApp("src/llmling_agent/config_resources/agents.yml")
    app.run()
