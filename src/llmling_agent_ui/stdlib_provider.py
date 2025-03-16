"""Stdlib UI Provider."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llmling_agent_ui.base import UIProvider


if TYPE_CHECKING:
    from llmling import ConfigStore

    from llmling_agent import AgentPool, MessageNode
    from llmling_agent_config.ui import StdlibUIConfig


class StdlibUIProvider(UIProvider):
    """Basic CLI interface using stdlib."""

    def __init__(self, config: StdlibUIConfig):
        self.config = config

    def run_pool(self, pool: AgentPool):
        """Run basic CLI interface."""
        import asyncio

        async def run_loop():
            def on_message(message):
                print(
                    message.format(
                        style=self.config.detail_level,
                        show_metadata=self.config.show_metadata,
                        show_costs=self.config.show_costs,
                    )
                )

            if self.config.show_messages:
                for agent in pool.agents.values():
                    agent.message_sent.connect(on_message)

            await pool.run_event_loop()

        asyncio.run(run_loop())

    def run_node(self, node: MessageNode):
        msg = "StdLibUI only supports pool mode"
        raise NotImplementedError(msg)

    def run(self, store: ConfigStore | None = None):
        msg = "StdLibUI only supports pool mode"
        raise NotImplementedError(msg)
