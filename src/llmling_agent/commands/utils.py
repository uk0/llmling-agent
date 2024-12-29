from __future__ import annotations

from typing import TYPE_CHECKING
import webbrowser

from slashed import Command, CommandContext, CommandError

from llmling_agent.log import get_logger
from llmling_agent.pydantic_ai_utils import find_last_assistant_message


if TYPE_CHECKING:
    from llmling_agent.chat_session.base import AgentChatSession


logger = get_logger(__name__)


EDIT_AGENT_HELP = """\
Open the agent's configuration file in your default editor.

This file contains:
- Agent settings and capabilities
- System prompts
- Model configuration
- Environment references
- Role definitions

Note: Changes to the configuration file require reloading the agent.

"""

COPY_CB_HELP = """\
"Copy the most recent assistant response to the system clipboard.\n"
"Requires pyperclip package to be installed."
"""


async def copy_clipboard(
    ctx: CommandContext[AgentChatSession],
    args: list[str],
    kwargs: dict[str, str],
):
    """Copy last assistant message to clipboard."""
    try:
        import pyperclip
    except ImportError as e:
        msg = "pyperclip package required for clipboard operations"
        raise CommandError(msg) from e

    if not ctx.context.history:
        await ctx.output.print("No messages to copy")
        return

    if content := find_last_assistant_message(ctx.context.history):
        try:
            pyperclip.copy(content)
            await ctx.output.print("Last assistant message copied to clipboard")
        except Exception as e:
            msg = f"Failed to copy to clipboard: {e}"
            raise CommandError(msg) from e
    else:
        await ctx.output.print("No assistant message found to copy")


async def edit_agent_file(
    ctx: CommandContext[AgentChatSession],
    args: list[str],
    kwargs: dict[str, str],
):
    """Open agent's configuration file in default application."""
    if not ctx.context._agent._context:
        msg = "No agent context available"
        raise CommandError(msg)

    config = ctx.context._agent._context.config
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
    help_text=EDIT_AGENT_HELP,
    category="utils",
)

copy_clipboard_cmd = Command(
    name="copy-clipboard",
    description="Copy the last assistant message to clipboard",
    execute_func=copy_clipboard,
    help_text=COPY_CB_HELP,
    category="utils",
)
