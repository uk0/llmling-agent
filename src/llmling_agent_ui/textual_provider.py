"""Textual UI Provider."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llmling_agent_ui.base import UIProvider


if TYPE_CHECKING:
    from llmling.config.store import ConfigStore

    from llmling_agent import AgentPool, MessageNode
    from llmling_agent_config.ui import TextualUIConfig


class TextualUIProvider(UIProvider):
    """Terminal UI using Textual."""

    def __init__(self, config: TextualUIConfig):
        self.config = config

    def run_pool(self, pool: AgentPool):
        """Run Textual interface.

        Textual manages its own event loop internally, so we just
        need to create and run the app.
        """
        from llmling_textual.app import PoolApp

        # TODO: change TextualApp to take a pool
        app = PoolApp(pool)
        app.run()  # This blocks and handles its own event loop

    def run_node(self, node: MessageNode):
        msg = "TextualUI only supports pool mode"
        raise NotImplementedError(msg)

    def run(self, store: ConfigStore | None = None):
        msg = "TextualUI only supports pool mode"
        raise NotImplementedError(msg)
