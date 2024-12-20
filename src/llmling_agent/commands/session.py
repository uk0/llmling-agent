from __future__ import annotations

from typing import TYPE_CHECKING

from slashed import Command, CommandContext


if TYPE_CHECKING:
    from llmling_agent.chat_session.base import AgentChatSession


async def clear_command(
    ctx: CommandContext[AgentChatSession],
    args: list[str],
    kwargs: dict[str, str],
) -> None:
    """Clear chat history."""
    await ctx.data.clear()


async def reset_command(
    ctx: CommandContext[AgentChatSession],
    args: list[str],
    kwargs: dict[str, str],
) -> None:
    """Reset session state."""
    await ctx.data.reset()


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
