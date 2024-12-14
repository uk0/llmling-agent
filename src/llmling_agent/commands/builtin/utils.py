from __future__ import annotations

from typing import TYPE_CHECKING
import webbrowser

from pydantic_ai.messages import Message, ModelStructuredResponse, ModelTextResponse

from llmling_agent.commands.base import Command, CommandContext, CommandError
from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from collections.abc import Sequence


logger = get_logger(__name__)


def find_last_assistant_message(messages: Sequence[Message]) -> str | None:
    """Find the last assistant message in history."""
    for msg in reversed(messages):
        match msg:
            case ModelTextResponse():
                return msg.content
            case ModelStructuredResponse():
                # Format structured response in a readable way
                calls = []
                for call in msg.calls:
                    if isinstance(call.args, dict):
                        args = call.args
                    else:
                        # Handle both ArgsJson and ArgsDict
                        args = (
                            call.args.args_dict
                            if hasattr(call.args, "args_dict")
                            else call.args.args_json
                        )
                    calls.append(f"Tool: {call.tool_name}\nArgs: {args}")
                return "\n\n".join(calls)
    return None


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


async def edit_env(
    ctx: CommandContext,
    args: list[str],
    kwargs: dict[str, str],
) -> None:
    """Open agent's environment file in default application."""
    if not ctx.session._agent._context:
        msg = "No agent context available"
        raise CommandError(msg)

    config = ctx.session._agent._context.config
    try:
        resolved_path = config.get_environment_path()
        webbrowser.open(resolved_path)
        await ctx.output.print(f"Opening environment file: {resolved_path}")
    except Exception as e:
        msg = f"Failed to open environment file: {e}"
        raise CommandError(msg) from e


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

edit_env_cmd = Command(
    name="edit-env",
    description="Edit the agent's environment configuration",
    execute_func=edit_env,
    help_text=(
        "Open the agent's environment configuration file in the default editor.\n"
        "This allows you to modify:\n"
        "- Available tools\n"
        "- Resources\n"
        "- Other environment settings"
    ),
    category="utils",
)
