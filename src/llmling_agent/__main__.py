"""CLI interface for LLMling agents."""

from __future__ import annotations

from llmling.cli.utils import get_command_help
import typer as t

from llmling_agent_cli.agent import add_agent_file, list_agents, set_active_file
from llmling_agent_cli.chat import chat_command
from llmling_agent_cli.history import history_cli
from llmling_agent_cli.quickstart import quickstart_command
from llmling_agent_cli.run import run_command
from llmling_agent_cli.serve import serve_command
from llmling_agent_cli.task import task_command
from llmling_agent_cli.ui import ui_command
from llmling_agent_cli.watch import watch_command


MAIN_HELP = "🤖 LLMling Agent CLI - Run and manage LLM agents"
MISSING_WEB = """
Web interface commands require gradio.
Install with: pip install llmling-agent[ui]
"""
# Create CLI app
help_text = get_command_help(MAIN_HELP)
cli = t.Typer(name="LLMling Agent", help=help_text, no_args_is_help=True)

cli.command(name="add")(add_agent_file)
cli.command(name="run")(run_command)
cli.command(name="list")(list_agents)
cli.command(name="set")(set_active_file)
cli.command(name="chat")(chat_command)
cli.command(name="quickstart")(quickstart_command)
cli.command(name="watch")(watch_command)
cli.command(name="serve")(serve_command)
cli.command(name="task")(task_command)
cli.command(name="ui")(ui_command)

cli.add_typer(history_cli, name="history")

try:
    from llmling_agent_cli import web

    cli.command(name="launch")(web.launch_gui)

except ImportError:
    web_cli = t.Typer(help="Web interface commands (not installed)")

    @web_cli.callback()
    def web_not_installed():
        """Web interface functionality (not installed)."""
        print(MISSING_WEB)
        raise t.Exit(1)

    cli.add_typer(web_cli, name="web")
