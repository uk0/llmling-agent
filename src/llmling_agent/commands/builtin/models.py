"""Model-related commands."""

from __future__ import annotations

from llmling_agent.commands.base import Command, CommandContext


async def set_model(
    ctx: CommandContext,
    args: list[str],
    kwargs: dict[str, str],
) -> None:
    """Change the model for the current conversation."""
    if not args:
        await ctx.output.print("Usage: /set-model <model>\nExample: /set-model gpt-4")
        return

    model = args[0]
    try:
        # Create new session with model override
        await ctx.session.reset()  # Clear history and reset state
        msg = "Model changed. How can I help you?"
        await ctx.session._agent.run(msg, model=model)  # type: ignore[arg-type]
        await ctx.output.print(f"Model changed to: {model}")
    except Exception as e:  # noqa: BLE001
        await ctx.output.print(f"Failed to change model: {e}")


set_model_cmd = Command(
    name="set-model",
    description="Change the model for the current conversation",
    execute_func=set_model,
    usage="<model>",
    help_text=(
        "Change the language model for the current conversation.\n\n"
        "The model change takes effect immediately for all following messages. "
        "Previous messages and their context are preserved.\n\n"
        "Examples:\n"
        "  /set-model gpt-4\n"
        "  /set-model openai:gpt-4o-mini\n"
        "  /set-model claude-2\n\n"
        "Note: Available models depend on your configuration and API access."
    ),
    category="model",
)
