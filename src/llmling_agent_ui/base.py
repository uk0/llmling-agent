"""Base UI provider class."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol


if TYPE_CHECKING:
    from llmling import ConfigStore

    from llmling_agent import AgentPool, MessageNode


class UIProvider(Protocol):
    """Protocol for UI providers."""

    def run_pool(self, pool: AgentPool):
        """Run the UI with a pool of agents."""
        raise NotImplementedError

    def run_node(self, node: MessageNode):
        """Run the UI with a single node (agent/team)."""
        raise NotImplementedError

    def run(self, store: ConfigStore | None = None):
        """Run the UI with access to the config store.

        This allows the UI to manage multiple pool configurations.
        """
        raise NotImplementedError
