"""Environment management commands."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
import webbrowser

from llmling import RuntimeConfig
from slashed import Command, CommandContext, CommandError, PathCompleter
from upath import UPath

from llmling_agent.agent.agent import Agent
from llmling_agent.agent.structured import StructuredAgent
from llmling_agent.environment.models import FileEnvironment, InlineEnvironment


if TYPE_CHECKING:
    from llmling_agent.agent import AnyAgent
    from llmling_agent.chat_session.base import AgentPoolView


SET_ENV_HELP = """\
Change the environment configuration file for the current session.

The environment file defines:
- Available tools
- Resource configurations
- Other runtime settings

Example: /set-env configs/new_env.yml

Note: This will reload the runtime configuration and update available tools.
"""

EDIT_ENV_HELP = """\
Open the agent's environment configuration file in the default editor.
This allows you to modify:
- Available tools
- Resources
- Other environment settings
"""


async def set_env(
    ctx: CommandContext[AgentPoolView],
    args: list[str],
    kwargs: dict[str, str],
):
    """Change the environment file path."""
    if not args:
        await ctx.output.print("Usage: /set-env <path>")
        return

    env_path = args[0]
    if not UPath(env_path).exists():
        msg = f"Environment file not found: {env_path}"
        raise CommandError(msg)

    try:
        # Get current agent configuration
        agent = ctx.context._agent
        if not agent.context.config:
            msg = "No agent context available"
            raise CommandError(msg)  # noqa: TRY301

        # Update environment path in config
        config = agent.context.config
        config = config.model_copy(update={"environment": env_path})

        # Create new runtime with updated config
        async with RuntimeConfig.open(config.get_config()) as new_runtime:
            # Create new agent with updated runtime
            kw_args = agent.context.config.get_agent_kwargs()
            new_agent: AnyAgent[Any, Any] = Agent(
                runtime=new_runtime, context=agent.context, **kw_args
            )
            if isinstance(agent, StructuredAgent):
                new_agent = new_agent.to_structured(
                    result_type=agent.result_type,
                    # tool_name=agent.tool_name,
                    # tool_description=agent.tool_description,
                )
            # Update session's agent
            ctx.context._agent = new_agent

            await ctx.output.print(
                f"Environment changed to: {env_path}\n"
                "Session updated with new configuration."
            )

    except Exception as e:
        msg = f"Failed to change environment: {e}"
        raise CommandError(msg) from e


async def edit_env(
    ctx: CommandContext[AgentPoolView],
    args: list[str],
    kwargs: dict[str, str],
):
    """Open agent's environment file in default application."""
    if not ctx.context._agent.context:
        msg = "No agent context available"
        raise CommandError(msg)

    config = ctx.context._agent.context.config
    match config.environment:
        case FileEnvironment(uri=uri):
            # For file environments, open in browser
            try:
                webbrowser.open(uri)
                await ctx.output.print(f"Opening environment file: {uri}")
            except Exception as e:
                msg = f"Failed to open environment file: {e}"
                raise CommandError(msg) from e
        case InlineEnvironment() as cfg:
            # For inline environments, display the configuration
            await ctx.output.print("Inline environment configuration:")
            yaml_config = cfg.model_dump_yaml()
            await ctx.output.print(yaml_config)
        case str() as path:
            # Legacy string path
            try:
                resolved = config._resolve_environment_path(path, config.config_file_path)
                webbrowser.open(resolved)
                await ctx.output.print(f"Opening environment file: {resolved}")
            except Exception as e:
                msg = f"Failed to open environment file: {e}"
                raise CommandError(msg) from e
        case None:
            await ctx.output.print("No environment configured")


set_env_cmd = Command(
    name="set-env",
    description="Change the environment configuration file",
    execute_func=set_env,
    usage="<path>",
    help_text=SET_ENV_HELP,
    category="environment",
    completer=PathCompleter(file_patterns=["*.yml", "*.yaml"]),
)

edit_env_cmd = Command(
    name="open-env-file",
    description="Open the agent's environment configuration",
    execute_func=edit_env,
    help_text=EDIT_ENV_HELP,
    category="environment",
    completer=PathCompleter(file_patterns=["*.yml", "*.yaml"]),
)
