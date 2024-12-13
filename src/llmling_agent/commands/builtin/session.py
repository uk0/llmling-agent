from __future__ import annotations

from llmling_agent.commands.base import Command, CommandContext


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


clear_cmd = Command(
    name="clear",
    description="Clear chat history",
    execute_func=clear_command,
    category="session",
)

reset_cmd = Command(
    name="reset",
    description="Reset session state (clear history and reset tools)",
    execute_func=reset_command,
    category="session",
)
