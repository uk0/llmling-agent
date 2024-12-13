"""Agent management commands."""

from __future__ import annotations

from llmling_agent.agent import LLMlingAgent
from llmling_agent.commands.base import Command, CommandContext


async def list_agents(
    ctx: CommandContext,
    args: list[str],
    kwargs: dict[str, str],
) -> None:
    """List all available agents."""
    # Get agent definition through context
    definition = ctx.session._agent._context.definition

    await ctx.output.print("\nAvailable agents:")
    for name, agent in definition.agents.items():
        desc = f" - {agent.description}" if agent.description else ""
        model = f" ({agent.model})" if agent.model else ""
        await ctx.output.print(f"  {name}{model}{desc}")


async def switch_agent(
    ctx: CommandContext,
    args: list[str],
    kwargs: dict[str, str],
) -> None:
    """Switch to a different agent."""
    if not args:
        await ctx.output.print("Usage: /switch-agent <name>")
        return

    name = args[0]
    definition = ctx.session._agent._context.definition

    if name not in definition.agents:
        await ctx.output.print(f"Unknown agent: {name}")
        return

    try:
        async with LLMlingAgent[str].open_agent(definition, name) as new_agent:
            # Update session's agent
            ctx.session._agent = new_agent
            # Reset session state
            ctx.session._history = []
            ctx.session._tool_states = new_agent.list_tools()
            await ctx.output.print(f"Switched to agent: {name}")
    except Exception as e:  # noqa: BLE001
        await ctx.output.print(f"Failed to switch agent: {e}")


list_agents_cmd = Command(
    name="list-agents",
    description="List available agents",
    execute_func=list_agents,
    category="agents",
)

switch_agent_cmd = Command(
    name="switch-agent",
    description="Switch to a different agent",
    execute_func=switch_agent,
    usage="<name>",
    help_text="Switch to a different agent by name.",
    category="agents",
)
