from __future__ import annotations

from textual.app import App

from llmling_agent.delegation.pool import AgentPool
from llmling_textual.screens.main_screen import MainScreen


class AgentApp(App):
    """Main application."""

    def __init__(self, config_path: str):
        super().__init__()
        self.agent_pool: AgentPool | None = None
        self._config_path = config_path

    async def on_mount(self) -> None:
        """Initialize pool and mount main screen."""
        self.agent_pool = AgentPool(self._config_path)
        self.main_screen = MainScreen(self.agent_pool)
        await self.agent_pool.__aenter__()
        await self.push_screen(self.main_screen)

    async def on_unmount(self) -> None:
        """Cleanup pool."""
        if self.agent_pool:
            await self.agent_pool.__aexit__(None, None, None)


if __name__ == "__main__":
    # Example usage - path should be passed from outside
    app = AgentApp("src/llmling_agent/config_resources/agents.yml")
    app.run()
