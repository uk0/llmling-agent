"""Agent-related CLI commands."""

from __future__ import annotations

import asyncio

from llmling.cli.constants import output_format_opt, verbose_opt
from llmling.cli.utils import format_output
from pydantic import ValidationError
import typer as t

from llmling_agent.cli import agent_store
from llmling_agent.factory import create_agents_from_config
from llmling_agent.models import AgentDefinition


agent_cli = t.Typer(help="Agent management commands", no_args_is_help=True)


def resolve_agent_config(config: str | None) -> str:
    """Resolve agent configuration path from name or direct path.

    Args:
        config: Configuration name or path

    Returns:
        Resolved configuration path

    Raises:
        ValueError: If no configuration is found
    """
    if not config:
        if active := agent_store.get_active():
            return active.path
        msg = "No active agent configuration set. Use 'agent set' to set one."
        raise ValueError(msg)

    try:
        # First try as stored config name
        return agent_store.get_config(config)
    except KeyError:
        # If not found, treat as direct path
        return config


@agent_cli.command("add")
def add_agent_file(
    name: str = t.Argument(help="Name for the agent configuration file"),
    path: str = t.Argument(help="Path to agent configuration file"),
    verbose: bool = verbose_opt,
) -> None:
    """Add a new agent configuration file."""
    try:
        agent_store.add_config(name, path)
        t.echo(f"Added agent configuration '{name}' -> {path}")
    except Exception as e:
        t.echo(f"Error adding configuration: {e}", err=True)
        raise t.Exit(1) from e


@agent_cli.command("set")
def set_active_file(
    name: str = t.Argument(help="Name of agent configuration to set as active"),
    verbose: bool = verbose_opt,
) -> None:
    """Set the active agent configuration file."""
    try:
        agent_store.set_active(name)
        t.echo(f"Set '{name}' as active agent configuration")
    except Exception as e:
        t.echo(f"Error setting active configuration: {e}", err=True)
        raise t.Exit(1) from e


@agent_cli.command("run")
def run_agent(
    agent_name: str = t.Argument(help="Name of the agent to run"),
    prompts: list[str] = t.Argument(  # noqa: B008
        None,
        help="Prompts to send to the agent",
    ),
    config_path: str = t.Option(
        None,
        "-c",
        "--config",
        help="Override agent configuration path",
    ),
    prompt_index: int = t.Option(
        0,
        "--index",
        "-i",
        help="Index of predefined prompt to start with",
    ),
    environment: str = t.Option(
        None,
        "--environment",
        "-e",
        help="Override agent's environment",
    ),
    model: str = t.Option(
        None,
        "--model",
        "-m",
        help="Override agent's model",
    ),
    verbose: bool = verbose_opt,
) -> None:
    """Run an agent with the given prompts."""
    try:
        # Resolve configuration path
        try:
            config_path = resolve_agent_config(config_path)
        except ValueError as e:
            msg = str(e)
            raise t.BadParameter(msg) from e

        try:
            # Load agent definition
            agent_def = AgentDefinition.from_file(config_path)
        except ValidationError as e:
            t.echo("Agent configuration validation failed:", err=True)
            for error in e.errors():
                location = " -> ".join(str(loc) for loc in error["loc"])
                t.echo(f"  {location}: {error['msg']}", err=True)
            raise t.Exit(1) from e

        # Check if agent exists
        if agent_name not in agent_def.agents:
            msg = f"Agent '{agent_name}' not found in configuration"
            raise t.BadParameter(msg)  # noqa: TRY301

        agent_config = agent_def.agents[agent_name]

        # Build final prompt list
        final_prompts = []

        # Start with predefined prompt if index given and prompts exist
        if agent_config.user_prompts:
            try:
                final_prompts.append(agent_config.user_prompts[prompt_index])
            except IndexError:
                num_prompts = len(agent_config.user_prompts) - 1
                msg = f"Prompt index {prompt_index} out of range (0-{num_prompts})"
                raise t.BadParameter(msg)  # noqa: B904

        # Append additional prompts
        if prompts:
            final_prompts.extend(prompts)

        if not final_prompts:
            msg = "No prompts provided and no default prompts in configuration"
            raise t.BadParameter(msg)  # noqa: TRY301

        # Apply overrides
        if model:
            agent_config.model = model

        async def _run() -> None:
            # Use CLI override or resolved path from config
            env_path = environment or agent_config.environment
            from llmling import Config
            from llmling.config.runtime import RuntimeConfig

            async with RuntimeConfig.open(env_path or Config()) as runtime:
                agents = create_agents_from_config(agent_def, runtime)
                agent = agents[agent_name]

                # Execute prompts as conversation
                result = await agent.run(final_prompts[0])
                format_output(result.data)

                for prompt in final_prompts[1:]:
                    result = await agent.run(
                        prompt,
                        message_history=result.new_messages(),
                    )
                    format_output(result.data)

        asyncio.run(_run())

    except t.Exit:
        raise
    except Exception as e:
        t.echo(f"Error: {e}", err=True)
        if verbose:
            import traceback

            t.echo(traceback.format_exc(), err=True)
        raise t.Exit(1) from e


@agent_cli.command("list")
def list_agents(
    config_name: str = t.Option(
        None,
        "-c",
        "--config",
        help="Name of agent configuration to list (defaults to active)",
    ),
    output_format: str = output_format_opt,
    verbose: bool = verbose_opt,
) -> None:
    """List agents from the active (or specified) configuration."""
    try:
        try:
            config_path = resolve_agent_config(config_name)
        except ValueError as e:
            msg = str(e)
            raise t.BadParameter(msg) from e

        try:
            # Load and validate agent definition
            agent_def = AgentDefinition.from_file(config_path)
            # Set the name field from the dict key for each agent
            agents = [
                agent.model_copy(update={"name": name})
                for name, agent in agent_def.agents.items()
            ]
            format_output(agents, output_format)

        except ValidationError as e:
            t.echo("Configuration validation failed:", err=True)
            for error in e.errors():
                location = " -> ".join(str(loc) for loc in error["loc"])
                t.echo(f"  {location}: {error['msg']}", err=True)
            raise t.Exit(1) from e

    except t.Exit:
        raise
    except Exception as e:
        t.echo(f"Error: {e}", err=True)
        if verbose:
            import traceback

            t.echo(traceback.format_exc(), err=True)
        raise t.Exit(1) from e
