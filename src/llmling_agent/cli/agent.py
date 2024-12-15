"""Agent-related CLI commands."""

from __future__ import annotations

import shutil

from llmling.cli.constants import output_format_opt, verbose_opt
from llmling.cli.utils import format_output
from promptantic import ModelGenerator
import typer as t
from upath import UPath

from llmling_agent import config_resources
from llmling_agent.cli import agent_store, resolve_agent_config
from llmling_agent.models import AgentsManifest


agent_cli = t.Typer(help="Agent management commands", no_args_is_help=True)


@agent_cli.command("init")
def init_agent_config(
    output: str = t.Argument(help="Path to write agent configuration file"),
    name: str | None = t.Option(
        None,
        "--name",
        "-n",
        help="Name for the configuration (defaults to filename)",
    ),
    interactive: bool = t.Option(
        False,
        "--interactive/--no-interactive",
        help="Use interactive configuration wizard",
    ),
) -> None:
    """Initialize a new agent configuration file.

    Creates and activates a new agent configuration. The configuration will be
    automatically registered and set as active.
    """
    # Create config
    if interactive:
        generator = ModelGenerator()
        manifest = generator.populate(AgentsManifest)
        manifest.save(output)
    else:
        shutil.copy2(config_resources.AGENTS_TEMPLATE, output)

    # Add and set as active
    config_name = name or UPath(output).stem
    agent_store.add_config(config_name, output)
    agent_store.set_active(config_name)

    print(f"\nCreated and activated agent configuration '{config_name}': {output}")
    print("\nTry these commands:")
    print("  llmling-agent list")
    print("  llmling-agent chat simple_agent")


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
    from llmling_agent.models import AgentsManifest

    try:
        try:
            config_path = resolve_agent_config(config_name)
        except ValueError as e:
            msg = str(e)
            raise t.BadParameter(msg) from e

        # Load and validate agent definition
        agent_def = AgentsManifest.from_file(config_path)
        # Set the name field from the dict key for each agent
        agents = [
            agent.model_copy(update={"name": name})
            for name, agent in agent_def.agents.items()
        ]
        format_output(agents, output_format)

    except t.Exit:
        raise
    except Exception as e:
        t.echo(f"Error: {e}", err=True)
        if verbose:
            import traceback

            t.echo(traceback.format_exc(), err=True)
        raise t.Exit(1) from e
