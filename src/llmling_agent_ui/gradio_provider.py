"""Gradio UI provider."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llmling_agent_ui.base import UIProvider


if TYPE_CHECKING:
    from llmling import ConfigStore

    from llmling_agent import AgentPool, MessageNode
    from llmling_agent_config.ui import GradioUIConfig


class GradioUIProvider(UIProvider):
    """Web interface using Gradio."""

    def __init__(self, config: GradioUIConfig):
        self.config = config

    def run(self, config_store: ConfigStore | None = None):
        """Run web interface."""
        from llmling_agent_web.app import launch_app

        launch_app(
            server_name=self.config.host,
            server_port=self.config.port,
            share=self.config.share,
            theme=self.config.theme,
            block=True,
        )

    def run_node(self, node: MessageNode):
        msg = "GradioUI only supports config store mode"
        raise NotImplementedError(msg)

    def run_pool(self, pool: AgentPool):
        msg = "GradioUI only supports config store mode"
        raise NotImplementedError(msg)
