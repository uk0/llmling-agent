"""CLI interface for LLMling agents."""

from __future__ import annotations

from llmling.cli.utils import get_command_help
import typer as t

from llmling_agent.cli.agent import (
    add_agent_file,
    list_agents,
    run_agent,
    set_active_file,
)


MAIN_HELP = "🤖 LLMling Agent CLI - Run and manage LLM agents"
MISSING_WEB = """
Web interface commands require gradio.
Install with: pip install llmling-agent[ui]
"""
# Create CLI app
cli = t.Typer(
    name="LLMling Agent",
    help=get_command_help(MAIN_HELP),
    no_args_is_help=True,
)

cli.command(name="add")(add_agent_file)
cli.command(name="run")(run_agent)
cli.command(name="list")(list_agents)
cli.command(name="set")(set_active_file)

try:
    from llmling_agent.cli.web import web_cli

    cli.add_typer(web_cli, name="web", help="Web interface commands")
except ImportError:
    web_cli = t.Typer(help="Web interface commands (not installed)")

    @web_cli.callback()
    def web_not_installed() -> None:
        """Web interface functionality (not installed)."""
        print(MISSING_WEB)
        raise t.Exit(1)

    cli.add_typer(web_cli, name="web")
