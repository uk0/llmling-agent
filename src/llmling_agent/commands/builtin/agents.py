"""Agent management commands."""

from __future__ import annotations

from typing import Any

import yaml

from llmling_agent.agent import LLMlingAgent
from llmling_agent.commands.base import Command, CommandContext


def create_annotated_dump(
    config: dict[str, Any],
    overrides: dict[str, Any],
    *,
    indent: int = 2,
) -> str:
    """Create YAML dump with override annotations.

    Args:
        config: Configuration dictionary
        overrides: Dictionary of overridden values
        indent: Indentation level (default: 2)

    Returns:
        YAML string with override comments
    """
    lines = []
    base_yaml = yaml.dump(
        config,
        sort_keys=False,
        indent=indent,
        default_flow_style=False,
        allow_unicode=True,
    )

    for line in base_yaml.splitlines():
        # Check if this line contains an overridden field
        for key in overrides:
            if line.startswith(f"{key}:"):
                # Add comment indicating original value
                original = config.get(key, "not set")
                lines.append(f"{line}  # Override (was: {original})")
                break
        else:
            lines.append(line)

    return "\n".join(lines)


async def show_agent(
    ctx: CommandContext,
    args: list[str],
    kwargs: dict[str, str],
) -> None:
    """Show current agent's configuration."""
    if not ctx.session._agent._context:
        await ctx.output.print("No agent context available")
        return

    # Get the agent's config with current overrides
    config = ctx.session._agent._context.config

    # Track overrides
    overrides = {}

    # Check model override
    if ctx.session._model:
        overrides["model"] = ctx.session._model

    # Get base config as dict
    config_dict = config.model_dump(exclude_none=True)

    # Apply overrides
    if overrides:
        config_dict.update(overrides)

    # Format as annotated YAML
    yaml_config = create_annotated_dump(config_dict, overrides)

    # Add header and format for display
    sections = [
        "\n[bold]Current Agent Configuration:[/]",
        "```yaml",
        yaml_config,
        "```",
    ]

    if overrides:
        sections.extend([
            "",
            "[dim]Note: Fields marked with '# Override' show runtime overrides[/]",
        ])

    await ctx.output.print("\n".join(sections))


show_agent_cmd = Command(
    name="show-agent",
    description="Show current agent's configuration",
    execute_func=show_agent,
    help_text=(
        "Display the complete configuration of the current agent as YAML.\n"
        "Shows:\n"
        "- Basic agent settings\n"
        "- Model configuration (with override indicators)\n"
        "- Environment settings (including inline environments)\n"
        "- System prompts\n"
        "- Response type configuration\n"
        "- Other settings\n\n"
        "Fields that have been overridden at runtime are marked with comments."
    ),
    category="agents",
)


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
            ctx.session._tool_states = new_agent.tools.list_tools()
            await ctx.output.print(f"Switched to agent: {name}")
    except Exception as e:  # noqa: BLE001
        await ctx.output.print(f"Failed to switch agent: {e}")


show_agent_cmd = Command(
    name="show-agent",
    description="Show current agent's configuration",
    execute_func=show_agent,
    help_text=(
        "Display detailed information about the current agent including:\n"
        "- Basic configuration\n"
        "- Model settings\n"
        "- Environment configuration\n"
        "- Response type details\n"
        "- System prompts\n"
        "- Other settings"
    ),
    category="agents",
)

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
