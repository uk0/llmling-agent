"""Slashed command to list nodes."""

from __future__ import annotations

from slashed import CommandContext, SlashedCommand

from llmling_agent.log import get_logger
from llmling_agent.messaging.context import NodeContext  # noqa: TC001


logger = get_logger(__name__)


class ListNodesCommand(SlashedCommand):
    """List all nodes in the pool with their status."""

    name = "list-nodes"
    category = "pool"

    async def execute_command(
        self,
        ctx: CommandContext[NodeContext],
        show_connections: bool = False,
    ):
        """List all nodes and their current status.

        Args:
            ctx: Command context with node
            show_connections: Whether to show node connections
        """
        node = ctx.get_data()
        header = "\nAvailable Nodes:"
        lines = [header, "=" * len(header)]
        assert node.pool
        for name, node_ in node.pool.nodes.items():
            # Build status info
            status = "🔄 busy" if node_.is_busy() else "⏳ idle"

            # Add connections if requested
            connections = []
            if show_connections and node_.connections.get_targets():
                connections = [a.name for a in node_.connections.get_targets()]
                conn_str = f" → {', '.join(connections)}"
            else:
                conn_str = ""

            # Add description if available
            desc = f" - {node_.description}" if node_.description else ""
            lines.append(f"{name} ({status}){conn_str}{desc}")
        await ctx.output.print("\n".join(lines))
