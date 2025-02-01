"""Model-related commands."""

from __future__ import annotations

from typing import TYPE_CHECKING

from slashed import Command, CommandContext
from slashed.completers import CallbackCompleter

from llmling_agent_commands.completers import get_model_names


if TYPE_CHECKING:
    from llmling_agent.agent.context import AgentContext


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
    ctx: CommandContext[AgentContext],
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
        ctx.context.agent.set_model(model)
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
    completer=CallbackCompleter(get_model_names),
)
