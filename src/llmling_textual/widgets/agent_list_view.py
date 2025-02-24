"""Agent list view."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.widgets import DataTable


if TYPE_CHECKING:
    from llmling_agent import AgentPool


class NodeListView(DataTable):
    """Display agents with their status and connections."""

    def __init__(self):
        super().__init__()
        self.add_columns("Name", "Status", "Model", "Connections")

    def update_agents(self, pool: AgentPool):
        self.clear()
        for name in pool.agents:
            agent = pool.get_agent(name)
            status = "🔄 busy" if agent.is_busy() else "⏳ idle"
            connections = [a.name for a in agent.connections.get_targets()]
            self.add_row(
                name, status, agent.model_name or "default", ", ".join(connections) or "-"
            )


if __name__ == "__main__":
    from textualicious import show

    show(NodeListView())
