"""Model-related commands."""

from __future__ import annotations

from typing import TYPE_CHECKING

from slashed import Command, CommandContext


if TYPE_CHECKING:
    from llmling_agent.chat_session.base import AgentChatSession


SET_MODEL_HELP = """\
Change the language model for the current conversation.

The model change takes effect immediately for all following messages.
Previous messages and their context are preserved.

Examples:
  /set-model gpt-4
  /set-model openai:gpt-4o-mini
  /set-model claude-2

Note: Available models depend on your configuration and API access.
"""


async def set_model(
    ctx: CommandContext[AgentChatSession],
    args: list[str],
    kwargs: dict[str, str],
):
    """Change the model for the current conversation."""
    if not args:
        await ctx.output.print("Usage: /set-model <model>\nExample: /set-model gpt-4")
        return

    model = args[0]
    try:
        # Create new session with model override
        ctx.data._agent.set_model(model)  # type: ignore
        # Update session's model override
        ctx.data._model = model
        await ctx.output.print(f"Model changed to: {model}")
    except Exception as e:  # noqa: BLE001
        await ctx.output.print(f"Failed to change model: {e}")


set_model_cmd = Command(
    name="set-model",
    description="Change the model for the current conversation",
    execute_func=set_model,
    usage="<model>",
    help_text=SET_MODEL_HELP,
    category="model",
)
