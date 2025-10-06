"""Agent session slash commands."""

from __future__ import annotations

from slashed import Command, CommandContext  # noqa: TC002

from llmling_agent.agent.context import AgentContext  # noqa: TC001


RESET_HELP = """\
Reset the entire session state:
- Clears chat history
- Restores default tool settings
- Resets any session-specific configurations
"""

CLEAR_HELP = """\
Clear the current chat session history.
This removes all previous messages but keeps tools and settings.
"""


async def clear_command(
    ctx: CommandContext[AgentContext],
    args: list[str],
    kwargs: dict[str, str],
):
    """Clear chat history."""
    ctx.context.agent.conversation.clear()


async def reset_command(
    ctx: CommandContext[AgentContext],
    args: list[str],
    kwargs: dict[str, str],
):
    """Reset session state."""
    ctx.context.agent.reset()


clear_cmd = Command(
    name="clear",
    description="Clear chat history",
    execute_func=clear_command,
    help_text=CLEAR_HELP,
    category="session",
)

reset_cmd = Command(
    name="reset",
    description="Reset session state",
    execute_func=reset_command,
    help_text=RESET_HELP,
    category="session",
)
