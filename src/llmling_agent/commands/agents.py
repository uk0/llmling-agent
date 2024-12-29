"""Agent management commands."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from llmling import Config
from slashed import Command, CommandContext, CommandError
from slashed.completers import CallbackCompleter
import yamling

from llmling_agent.agent import LLMlingAgent
from llmling_agent.commands.completers import get_available_agents
from llmling_agent.environment.models import InlineEnvironment
from llmling_agent.models.agents import AgentConfig


if TYPE_CHECKING:
    from llmling_agent.chat_session.base import AgentChatSession


CREATE_AGENT_HELP = """\
Create a new agent in the current session.

Creates a temporary agent that inherits the current agent's model.
The new agent will exist only for this session.

Options:
  --system-prompt "prompt"   System instructions for the agent (required)
  --model model_name        Override model (default: same as current agent)
  --role role_name         Agent role (assistant/specialist/overseer)
  --description "text"     Optional description of the agent

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
    ctx: CommandContext[AgentChatSession],
    args: list[str],
    kwargs: dict[str, str],
):
    """Create a new agent in the current session."""
    if not args:
        await ctx.output.print("Usage: /create-agent <name> --system-prompt 'prompt'")
        return

    name = args[0]
    system_prompt = kwargs.get("system-prompt")
    if not system_prompt:
        await ctx.output.print("Error: --system-prompt is required")
        return

    try:
        if not ctx.context.pool:
            msg = "No agent pool available"
            raise CommandError(msg)

        # Copy current agent's configuration with modifications
        current_agent = ctx.context._agent
        model = kwargs.get("model") or current_agent.model_name

        config = AgentConfig(
            name=name,
            model=model,
            system_prompts=[system_prompt],
            role=kwargs.get("role", "basic"),
            description=kwargs.get("description"),
            environment=InlineEnvironment(config=Config()),
            # config_file_path=current_agent._context.config.config_file_path,
        )

        _agent = await ctx.context.pool.create_agent(name, config, temporary=True)

        if kwargs.get("model"):
            msg = f"Created agent '{name}' with model {model}"
        else:
            msg = f"Created agent '{name}' (using current model: {model})"
        await ctx.output.print(f"{msg}\nUse /connect {name} to forward messages")

    except ValueError as e:
        msg = f"Failed to create agent: {e}"
        raise CommandError(msg) from e


async def show_agent(
    ctx: CommandContext[AgentChatSession], args: list[str], kwargs: dict[str, str]
):
    """Show current agent's configuration."""
    if not ctx.context._agent._context:
        await ctx.output.print("No agent context available")
        return

    # Get the agent's config with current overrides
    config = ctx.context._agent._context.config

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
        "\n[bold]Current Agent Configuration:[/]",
        "```yaml",
        yaml_config,
        "```",
    ]

    await ctx.output.print("\n".join(sections))


async def list_agents(ctx: CommandContext, args: list[str], kwargs: dict[str, str]):
    """List all available agents."""
    # Get agent definition through context
    definition = ctx.context._agent._context.definition

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


async def switch_agent(ctx: CommandContext, args: list[str], kwargs: dict[str, str]):
    """Switch to a different agent."""
    if not args:
        await ctx.output.print("Usage: /switch-agent <name>")
        return

    name = args[0]
    definition = ctx.context._agent._context.definition

    if name not in definition.agents:
        await ctx.output.print(f"Unknown agent: {name}")
        return

    try:
        async with LLMlingAgent[Any, str].open_agent(definition, name) as new_agent:
            # Update session's agent
            ctx.context._agent = new_agent
            # Reset session state
            ctx.context._history = []
            ctx.context._tool_states = new_agent.tools.list_tools()
            await ctx.output.print(f"Switched to agent: {name}")
    except Exception as e:  # noqa: BLE001
        await ctx.output.print(f"Failed to switch agent: {e}")


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
