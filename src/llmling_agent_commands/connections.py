"""Agent connection management commands."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.tree import Tree
from slashed import Command, CommandContext, CommandError
from slashed.completers import CallbackCompleter

from llmling_agent.log import get_logger
from llmling_agent_commands.completers import get_available_agents


if TYPE_CHECKING:
    from llmling_agent.agent import AnyAgent
    from llmling_agent.chat_session.base import AgentPoolView


logger = get_logger(__name__)


CONNECT_HELP = """\
Connect the current agent to another agent.
Messages will be forwarded to the target agent.

Examples:
  /connect agent2          # Forward to agent, wait for responses
  /connect agent2 --no-wait  # Forward without waiting
"""

DISCONNECT_HELP = """\
Disconnect the current agent from a target agent.
Stops forwarding messages to the specified agent.

Example: /disconnect agent2
"""

LIST_CONNECTIONS_HELP = """\
Show current agent connections and their status.
Displays:
- Connected agents
- Wait settings
- Message flow direction
"""


def format_agent_name(agent: AnyAgent[Any, Any], current: bool = False) -> str:
    """Format agent name for display."""
    name = agent.name
    if current:
        return f"[bold blue]{name}[/]"
    if agent.connections.get_targets():
        return f"[green]{name}[/]"
    return f"[dim]{name}[/]"


async def connect_command(
    ctx: CommandContext[AgentPoolView],
    args: list[str],
    kwargs: dict[str, str],
):
    """Connect to another agent."""
    if not args:
        await ctx.output.print("Usage: /connect <agent_name> [--no-wait]")
        return

    target = args[0]
    wait = kwargs.get("wait", "true").lower() != "false"

    try:
        await ctx.context.connect_to(target, wait)
        msg = f"Now forwarding messages to {target}"
        msg += " (waiting for responses)" if wait else " (async)"
        await ctx.output.print(msg)
    except Exception as e:
        msg = f"Failed to connect to {target}: {e}"
        raise CommandError(msg) from e


async def disconnect_command(
    ctx: CommandContext[AgentPoolView],
    args: list[str],
    kwargs: dict[str, str],
):
    """Disconnect from an agent."""
    if not args:
        await ctx.output.print("Usage: /disconnect <agent_name>")
        return

    target = args[0]
    try:
        assert ctx.context.pool
        target_agent = ctx.context.pool.get_agent(target)
        ctx.context._agent.connections.disconnect(target_agent)
        await ctx.output.print(f"Stopped forwarding messages to {target}")
    except Exception as e:
        msg = f"Failed to disconnect from {target}: {e}"
        raise CommandError(msg) from e


async def disconnect_all_command(
    ctx: CommandContext[AgentPoolView],
    args: list[str],
    kwargs: dict[str, str],
):
    """Disconnect from all agents."""
    if not ctx.context._agent.connections.get_targets():
        await ctx.output.print("No active connections")
        return

    await ctx.context._agent.disconnect_all()
    await ctx.output.print("Disconnected from all agents")


async def list_connections(
    ctx: CommandContext[AgentPoolView],
    args: list[str],
    kwargs: dict[str, str],
):
    """List current connections."""
    if not ctx.context._agent.connections.get_targets():
        await ctx.output.print("No active connections")
        return

    # Create tree visualization
    tree = Tree(format_agent_name(ctx.context._agent, current=True))

    # Use session's get_connections() for info
    for agent in ctx.context._agent.connections.get_targets():
        assert ctx.context.pool
        name = format_agent_name(ctx.context.pool.get_agent(agent.name))
        _branch = tree.add(name)

    # Create string representation
    from rich.console import Console

    console = Console()
    with console.capture() as capture:
        console.print(tree)
    tree_str = capture.get()

    await ctx.output.print("\nConnection Tree:")
    await ctx.output.print(tree_str)


connect_cmd = Command(
    name="connect",
    description="Connect to another agent",
    execute_func=connect_command,
    usage="<agent_name> [--no-wait]",
    help_text=CONNECT_HELP,
    category="agents",
    completer=CallbackCompleter(get_available_agents),
)

disconnect_cmd = Command(
    name="disconnect",
    description="Disconnect from an agent",
    execute_func=disconnect_command,
    usage="<agent_name>",
    help_text=DISCONNECT_HELP,
    category="agents",
    completer=CallbackCompleter(get_available_agents),
)

disconnect_all_cmd = Command(
    name="disconnect-all",
    description="Disconnect from all agents",
    execute_func=disconnect_all_command,
    help_text="Remove all agent connections",
    category="agents",
)

connections_cmd = Command(
    name="connections",
    description="List current agent connections",
    execute_func=list_connections,
    help_text=LIST_CONNECTIONS_HELP,
    category="agents",
)
