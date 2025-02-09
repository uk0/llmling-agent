"""Agent connection management commands."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.tree import Tree
from slashed import Command, CommandContext, CommandError
from slashed.completers import CallbackCompleter

from llmling_agent.log import get_logger
from llmling_agent.messaging.context import NodeContext  # noqa: TC001
from llmling_agent.messaging.messagenode import MessageNode
from llmling_agent_commands.completers import get_available_nodes


if TYPE_CHECKING:
    from llmling_agent.messaging.messageemitter import MessageEmitter


logger = get_logger(__name__)


CONNECT_HELP = """\
Connect the current node to another node.
Messages will be forwarded to the target node.

Examples:
  /connect node2          # Forward to node, wait for responses
  /connect node2 --no-wait  # Forward without waiting
"""

DISCONNECT_HELP = """\
Disconnect the current node from a target node.
Stops forwarding messages to the specified node.

Example: /disconnect node2
"""

LIST_CONNECTIONS_HELP = """\
Show current node connections and their status.
Displays:
- Connected nodes
- Wait settings
- Message flow direction
"""


def format_node_name(node: MessageEmitter[Any, Any], current: bool = False) -> str:
    """Format node name for display."""
    name = node.name
    if current:
        return f"[bold blue]{name}[/]"
    if node.connections.get_targets():
        return f"[green]{name}[/]"
    return f"[dim]{name}[/]"


async def connect_command(
    ctx: CommandContext[NodeContext],
    args: list[str],
    kwargs: dict[str, str],
):
    """Connect to another node."""
    if not args:
        await ctx.output.print("Usage: /connect <node_name> [--no-wait]")
        return

    target = args[0]
    wait = kwargs.get("wait", "true").lower() != "false"
    source = ctx.context.node_name

    try:
        assert ctx.context.pool
        target_node = ctx.context.pool[target]
        assert isinstance(target_node, MessageNode)
        ctx.context.node.connect_to(target_node)
        ctx.context.node.connections.set_wait_state(
            target, wait if wait is not None else True
        )

        msg = f"{source!r} now forwarding messages to {target!r}"
        msg += " (waiting for responses)" if wait else " (async)"
        await ctx.output.print(msg)
    except Exception as e:
        msg = f"Failed to connect {source!r} to {target!r}: {e}"
        raise CommandError(msg) from e


async def disconnect_command(
    ctx: CommandContext[NodeContext],
    args: list[str],
    kwargs: dict[str, str],
):
    """Disconnect from another node."""
    if not args:
        await ctx.output.print("Usage: /disconnect <node_name>")
        return

    target = args[0]
    source = ctx.context.node_name
    try:
        assert ctx.context.pool
        target_node = ctx.context.pool[target]
        assert isinstance(target_node, MessageNode)
        ctx.context.node.connections.disconnect(target_node)
        await ctx.output.print(f"{source!r} stopped forwarding messages to {target!r}")
    except Exception as e:
        msg = f"{source!r} failed to disconnect from {target!r}: {e}"
        raise CommandError(msg) from e


async def disconnect_all_command(
    ctx: CommandContext[NodeContext],
    args: list[str],
    kwargs: dict[str, str],
):
    """Disconnect from all nodes."""
    if not ctx.context.node.connections.get_targets():
        await ctx.output.print("No active connections")
        return
    source = ctx.context.node_name
    await ctx.context.node.disconnect_all()
    await ctx.output.print(f"Disconnected {source!r} from all nodes")


async def list_connections(
    ctx: CommandContext[NodeContext],
    args: list[str],
    kwargs: dict[str, str],
):
    """List current connections."""
    if not ctx.context.node.connections.get_targets():
        await ctx.output.print("No active connections")
        return

    # Create tree visualization
    tree = Tree(format_node_name(ctx.context.node, current=True))

    # Use session's get_connections() for info
    for node in ctx.context.node.connections.get_targets():
        assert ctx.context.pool
        name = format_node_name(ctx.context.pool[node.name])
        _branch = tree.add(name)

    # Create string representation
    from rich.console import Console

    console = Console()
    with console.capture() as capture:
        console.print(tree)
    tree_str = capture.get()
    await ctx.output.print(f"\nConnection Tree:\n{tree_str}")


connect_cmd = Command(
    name="connect",
    description="Connect to another node",
    execute_func=connect_command,
    usage="<node_name> [--no-wait]",
    help_text=CONNECT_HELP,
    category="nodes",
    completer=CallbackCompleter(get_available_nodes),
)

disconnect_cmd = Command(
    name="disconnect",
    description="Disconnect from an node",
    execute_func=disconnect_command,
    usage="<node_name>",
    help_text=DISCONNECT_HELP,
    category="nodes",
    completer=CallbackCompleter(get_available_nodes),
)

disconnect_all_cmd = Command(
    name="disconnect-all",
    description="Disconnect from all nodes",
    execute_func=disconnect_all_command,
    help_text="Remove all node connections",
    category="nodes",
)

connections_cmd = Command(
    name="connections",
    description="List current node connections",
    execute_func=list_connections,
    help_text=LIST_CONNECTIONS_HELP,
    category="nodes",
)
