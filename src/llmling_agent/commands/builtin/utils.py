from __future__ import annotations

import webbrowser

from llmling_agent.commands.base import Command, CommandContext
from llmling_agent.commands.exceptions import CommandError
from llmling_agent.log import get_logger
from llmling_agent.pydantic_ai_utils import find_last_assistant_message


logger = get_logger(__name__)


async def copy_clipboard(
    ctx: CommandContext,
    args: list[str],
    kwargs: dict[str, str],
) -> None:
    """Copy last assistant message to clipboard."""
    try:
        import pyperclip
    except ImportError as e:
        msg = "pyperclip package required for clipboard operations"
        raise CommandError(msg) from e

    if not ctx.session.history:
        await ctx.output.print("No messages to copy")
        return

    if content := find_last_assistant_message(ctx.session.history):
        try:
            pyperclip.copy(content)
            await ctx.output.print("Last assistant message copied to clipboard")
        except Exception as e:
            msg = f"Failed to copy to clipboard: {e}"
            raise CommandError(msg) from e
    else:
        await ctx.output.print("No assistant message found to copy")


async def edit_agent_file(
    ctx: CommandContext,
    args: list[str],
    kwargs: dict[str, str],
) -> None:
    """Open agent's configuration file in default application."""
    if not ctx.session._agent._context:
        msg = "No agent context available"
        raise CommandError(msg)

    config = ctx.session._agent._context.config
    if not config.config_file_path:
        msg = "No configuration file path available"
        raise CommandError(msg)

    try:
        webbrowser.open(config.config_file_path)
        await ctx.output.print(f"Opening agent configuration: {config.config_file_path}")
    except Exception as e:
        msg = f"Failed to open configuration file: {e}"
        raise CommandError(msg) from e


edit_agent_file_cmd = Command(
    name="open-agent-file",
    description="Open the agent's configuration file",
    execute_func=edit_agent_file,
    help_text=(
        "Open the agent's configuration file in your default editor.\n\n"
        "This file contains:\n"
        "- Agent settings and capabilities\n"
        "- System prompts\n"
        "- Model configuration\n"
        "- Environment references\n"
        "- Role definitions\n\n"
        "Note: Changes to the configuration file require reloading the agent."
    ),
    category="utils",
)

copy_clipboard_cmd = Command(
    name="copy-clipboard",
    description="Copy the last assistant message to clipboard",
    execute_func=copy_clipboard,
    help_text=(
        "Copy the most recent assistant response to the system clipboard.\n"
        "Requires pyperclip package to be installed."
    ),
    category="utils",
)
