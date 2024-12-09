"""Agent-related CLI commands."""

from __future__ import annotations

import asyncio

from llmling import Config, RuntimeConfig
from llmling.cli.constants import verbose_opt
from llmling.cli.utils import format_output
from llmling.core import exceptions
from pydantic import ValidationError
import typer as t

from llmling_agent.cli import agent_store
from llmling_agent.factory import create_agents_from_config
from llmling_agent.models import AgentDefinition


agent_cli = t.Typer(help="Agent management commands", no_args_is_help=True)


@agent_cli.command("run")
def run_agent(
    agent_name: str = t.Argument(help="Name of the agent to run"),
    prompts: list[str] = t.Argument(  # noqa: B008
        None,
        help="Prompts to send to the agent",
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
        # Get active agent file
        if active := agent_store.get_active():
            agent_file = active.path
        else:
            msg = "No active agent configuration set"
            raise t.BadParameter(msg)  # noqa: TRY301

        try:
            # Load agent definition
            agent_def = AgentDefinition.from_file(agent_file)
        except ValidationError as e:
            # Pretty-print validation errors
            t.echo("Agent configuration validation failed:", err=True)
            for error in e.errors():
                location = " -> ".join(str(loc) for loc in error["loc"])
                t.echo(f"  {location}: {error['msg']}", err=True)
            raise t.Exit(1) from e
        except exceptions.ConfigError as e:
            # Handle LLMling config errors
            t.echo(f"Configuration error: {e}", err=True)
            raise t.Exit(1) from e
        except Exception as e:
            # Handle unexpected errors during loading
            t.echo(f"Failed to load agent configuration: {e}", err=True)
            if verbose:
                import traceback

                t.echo(traceback.format_exc(), err=True)
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
            try:
                # Use CLI override or resolved path from config
                env_path = environment or agent_config.environment
                async with RuntimeConfig.open(env_path or Config()) as runtime:
                    agents = create_agents_from_config(agent_def, runtime)
                    agent = agents[agent_name]

                    # Execute prompts as conversation
                    result = await agent.run(final_prompts[0])
                    format_output(result.data)

                    for prompt in final_prompts[1:]:
                        result = await agent.run(
                            prompt, message_history=result.new_messages()
                        )
                        format_output(result.data)

            except ValidationError as e:
                t.echo("Environment configuration validation failed:", err=True)
                for error in e.errors():
                    location = " -> ".join(str(loc) for loc in error["loc"])
                    t.echo(f"  {location}: {error['msg']}", err=True)
                raise t.Exit(1) from e
            except exceptions.ConfigError as e:
                t.echo(f"Environment configuration error: {e}", err=True)
                raise t.Exit(1) from e

        asyncio.run(_run())

    except t.Exit:
        raise
    except Exception as e:
        t.echo(f"Unexpected error: {e}", err=True)
        if verbose:
            import traceback

            t.echo(traceback.format_exc(), err=True)
        raise t.Exit(1) from e


@agent_cli.command("add")
def add_agent_file(
    name: str = t.Argument(help="Name for the agent configuration file"),
    path: str = t.Argument(help="Path to agent configuration file"),
) -> None:
    """Add a new agent configuration file."""
    try:
        agent_store.add_config(name, path)
        t.echo(f"Added agent configuration file '{name}' -> {path}")
    except Exception as e:
        t.echo(f"Error: {e}", err=True)
        raise t.Exit(1) from e


@agent_cli.command("set")
def set_active_file(
    name: str = t.Argument(help="Name of agent configuration file to set as active"),
) -> None:
    """Set the active agent configuration file."""
    try:
        agent_store.set_active(name)
        t.echo(f"Set '{name}' as active agent configuration file")
    except Exception as e:
        t.echo(f"Error: {e}", err=True)
        raise t.Exit(1) from e


@agent_cli.command("list")
def list_agents(
    output_format: str = t.Option(
        "text",
        "--output-format",
        "-o",
        help="Output format (text/json/yaml)",
    ),
    verbose: bool = verbose_opt,
) -> None:
    """List all registered agents."""
    try:
        agents = agent_store.list_configs()
        active = agent_store.get_active()

        # If we have agents, load and validate their configurations
        agent_info = []
        has_errors = False

        for name, path in agents:
            try:
                agent_def = AgentDefinition.from_file(path)
                info = {
                    "name": name,
                    "path": path,
                    "active": active and name == active.name,
                    "agents": [
                        {
                            "name": agent_name,
                            "model": agent.model,
                            "environment": agent.environment,
                            "result_type": agent.result_type,
                        }
                        for agent_name, agent in agent_def.agents.items()
                    ],
                }
                agent_info.append(info)
            except ValidationError as e:
                has_errors = True
                t.echo(f"\nValidation errors in {path}:", err=True)
                for error in e.errors():
                    location = " -> ".join(str(loc) for loc in error["loc"])
                    t.echo(f"  {location}: {error['msg']}", err=True)
            except Exception as e:  # noqa: BLE001
                has_errors = True
                t.echo(f"\nError loading {path}: {e}", err=True)
                if verbose:
                    import traceback

                    t.echo(traceback.format_exc(), err=True)

        if not agent_info:
            if not agents:
                t.echo("No agent configurations registered", err=True)
            raise t.Exit(1)  # noqa: TRY301

        if has_errors:
            t.echo("\nWarning: Some configurations had errors", err=True)

        format_output({"agents": agent_info}, output_format)

    except t.Exit:
        raise
    except Exception as e:  # noqa: BLE001
        t.echo(f"Error: {e}", err=True)
        if verbose:
            import traceback

            t.echo(traceback.format_exc(), err=True)
        raise t.Exit(1)  # noqa: B904
