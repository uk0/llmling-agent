from __future__ import annotations

from llmling_agent.commands.base import Command, CommandContext
from llmling_agent.commands.exceptions import ExitCommandError


async def clear_command(
    ctx: CommandContext,
    args: list[str],
    kwargs: dict[str, str],
) -> None:
    """Clear chat history."""
    await ctx.session.clear()


async def reset_command(
    ctx: CommandContext,
    args: list[str],
    kwargs: dict[str, str],
) -> None:
    """Reset session state."""
    await ctx.session.reset()


async def exit_command(
    ctx: CommandContext,
    args: list[str],
    kwargs: dict[str, str],
) -> None:
    """Exit the chat session."""
    msg = "Session ended."
    raise ExitCommandError(msg)


exit_cmd = Command(
    name="exit",
    description="Exit chat session",
    execute_func=exit_command,
    category="cli",
)

clear_cmd = Command(
    name="clear",
    description="Clear chat history",
    execute_func=clear_command,
    help_text=(
        "Clear the current chat session history.\n"
        "This removes all previous messages but keeps tools and settings."
    ),
    category="session",
)

reset_cmd = Command(
    name="reset",
    description="Reset session state",
    execute_func=reset_command,
    help_text=(
        "Reset the entire session state:\n"
        "- Clears chat history\n"
        "- Restores default tool settings\n"
        "- Resets any session-specific configurations"
    ),
    category="session",
)
