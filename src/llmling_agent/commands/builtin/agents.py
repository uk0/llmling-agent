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
        # Keep the name clean and prominent
        name_part = name

        # Keep extra info simple with consistent width
        model_part = str(agent.model) if agent.model else ""
        desc_part = agent.description if agent.description else ""
        env_part = f"📄 {agent.environment}" if agent.environment else ""

        # Use dim style but maintain alignment
        await ctx.output.print(
            f"  {name_part:<20}[dim]{model_part:<15}{desc_part:<30}{env_part}[/dim]"
        )


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
    help_text=(
        "Show all agents defined in the current configuration.\n"
        "Displays:\n"
        "- Agent name\n"
        "- Model used (if specified)\n"
        "- Description (if available)"
    ),
    category="agents",
)

switch_agent_cmd = Command(
    name="switch-agent",
    description="Switch to a different agent",
    execute_func=switch_agent,
    usage="<name>",
    help_text=(
        "Switch the current chat session to a different agent.\n"
        "Use /list-agents to see available agents.\n\n"
        "Example: /switch-agent url_opener"
    ),
    category="agents",
)
