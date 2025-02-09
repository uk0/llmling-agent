"""Agent management commands."""

from __future__ import annotations

from typing import TYPE_CHECKING

from slashed import Command, CommandContext, CommandError
from slashed.completers import CallbackCompleter

from llmling_agent_commands.completers import get_available_agents


if TYPE_CHECKING:
    from llmling_agent.agent.context import AgentContext
    from llmling_agent.messaging.context import NodeContext


CREATE_AGENT_HELP = """\
Create a new agent in the current session.

Creates a temporary agent that inherits the current agent's model.
The new agent will exist only for this session.

Options:
  --system-prompt "prompt"   System instructions for the agent (required)
  --model model_name        Override model (default: same as current agent)
  --role role_name         Agent role (assistant/specialist/overseer)
  --description "text"     Optional description of the agent
  --tools "import_path1|import_path2"   Optional list tools (by import path)

Examples:
  # Create poet using same model as current agent
  /create-agent poet --system-prompt "Create poems from any text"

  # Create analyzer with different model
  /create-agent analyzer --system-prompt "Analyze in detail" --model gpt-4

  # Create specialized helper
  /create-agent helper --system-prompt "Debug code" --role specialist
"""

SHOW_AGENT_HELP_TEXT = """\
Display the complete configuration of the current agent as YAML.
Shows:
- Basic agent settings
- Model configuration (with override indicators)
- Environment settings (including inline environments)
- System prompts
- Response type configuration
- Other settings

Fields that have been overridden at runtime are marked with comments.
"""

LIST_AGENTS_HELP = """\
Show all agents defined in the current configuration.
Displays:
- Agent name
- Model used (if specified)
- Description (if available)
"""

SWITCH_AGENT_HELP = """\
Switch the current chat session to a different agent.
Use /list-agents to see available agents.

Example: /switch-agent url_opener
"""


async def create_agent_command(
    ctx: CommandContext[AgentContext],
    args: list[str],
    kwargs: dict[str, str],
):
    """Create a new agent in the current session."""
    if not args:
        await ctx.output.print(
            "Usage: /create-agent <name> --system-prompt 'prompt' [--tools 'tool1|tool2']"
        )
        return

    name = args[0]
    try:
        if not ctx.context.pool:
            msg = "No agent pool available"
            raise CommandError(msg)

        # Get model from args or current agent
        current_agent = ctx.context.agent

        # Parse tools if provided
        tools = None
        if tool_str := kwargs.get("tools"):
            tools = [t.strip() for t in tool_str.split("|")]

        # Create and register the new agent
        await ctx.context.pool.add_agent(
            name=name,
            model=kwargs.get("model") or current_agent.model_name,
            system_prompt=kwargs.get("system-prompt") or (),
            description=kwargs.get("description"),
            tools=tools,
        )

        msg = f"Created agent '{name}'"
        if tools:
            msg += f" with tools: {', '.join(tools)}"
        await ctx.output.print(f"{msg}\nUse /connect {name} to forward messages")

    except ValueError as e:
        msg = f"Failed to create agent: {e}"
        raise CommandError(msg) from e


async def show_agent(
    ctx: CommandContext[NodeContext], args: list[str], kwargs: dict[str, str]
):
    """Show current agent's configuration."""
    import yamling

    if not ctx.context.node.context:
        await ctx.output.print("No node context available")
        return

    # Get the agent's config with current overrides
    config = ctx.context.node.context.config

    # Get base config as dict
    config_dict = config.model_dump(exclude_none=True)

    # Format as annotated YAML
    yaml_config = yamling.dump_yaml(
        config_dict,
        sort_keys=False,
        indent=2,
        default_flow_style=False,
        allow_unicode=True,
    )
    # Add header and format for display
    sections = [
        "\n[bold]Current Node Configuration:[/]",
        "```yaml",
        yaml_config,
        "```",
    ]

    await ctx.output.print("\n".join(sections))


async def list_agents(
    ctx: CommandContext[AgentContext],
    args: list[str],
    kwargs: dict[str, str],
):
    """List all available agents."""
    if not ctx.context.pool:
        msg = "No agent pool available"
        raise CommandError(msg)

    output_lines = ["\nAvailable agents:"]

    # Collect all agent info first
    for name, agent in ctx.context.pool.agents.items():
        name_part = name
        model_part = str(agent.model_name or "")
        desc_part = agent.description if agent.description else ""

        # For dynamically created ones, add indicator
        dynamic = (
            "[dim](dynamic)[/dim] " if name not in ctx.context.definition.agents else ""
        )

        # Format line with proper padding
        line = f"  {name_part:<20}{dynamic}[dim]{model_part:<15}{desc_part}[/dim]"
        output_lines.append(line)

    # Send all lines at once
    await ctx.output.print("\n".join(output_lines))


async def switch_agent(
    ctx: CommandContext[AgentContext],
    args: list[str],
    kwargs: dict[str, str],
):
    """Switch to a different agent."""
    msg = "Temporarily disabled"
    raise RuntimeError(msg)
    if not args:
        await ctx.output.print("Usage: /switch-agent <name>")
        return


create_agent_cmd = Command(
    name="create-agent",
    description="Create a new agent in the current session",
    execute_func=create_agent_command,
    usage="<name> --system-prompt 'prompt' [--model name] [--role name]",
    help_text=CREATE_AGENT_HELP,
    category="agents",
)

show_agent_cmd = Command(
    name="show-agent",
    description="Show current agent's configuration",
    execute_func=show_agent,
    help_text=SHOW_AGENT_HELP_TEXT,
    category="agents",
)

list_agents_cmd = Command(
    name="list-agents",
    description="List available agents",
    execute_func=list_agents,
    help_text=LIST_AGENTS_HELP,
    category="agents",
)

switch_agent_cmd = Command(
    name="switch-agent",
    description="Switch to a different agent",
    execute_func=switch_agent,
    usage="<name>",
    help_text=SWITCH_AGENT_HELP,
    category="agents",
    completer=CallbackCompleter(get_available_agents),
)
