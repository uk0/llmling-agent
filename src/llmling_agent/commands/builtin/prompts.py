"""Prompt-related commands."""

from __future__ import annotations

from llmling_agent.commands.base import Command, CommandContext


async def list_prompts(
    ctx: CommandContext,
    args: list[str],
    kwargs: dict[str, str],
) -> None:
    """List available prompts."""
    prompts = ctx.session._agent.runtime.get_prompts()
    await ctx.output.print("\nAvailable prompts:")
    for prompt in prompts:
        await ctx.output.print(f"  {prompt.name:<20} - {prompt.description}")


async def prompt_command(
    ctx: CommandContext,
    args: list[str],
    kwargs: dict[str, str],
) -> None:
    """Execute a prompt.

    The first argument is the prompt name, remaining kwargs are prompt arguments.
    """
    if not args:
        await ctx.output.print("Usage: /prompt <name> [arg1=value1] [arg2=value2]")
        return

    name = args[0]
    try:
        messages = await ctx.session._agent.runtime.render_prompt(name, kwargs)
        # Convert prompt messages to chat format and send to agent
        for msg in messages:
            # TODO: Handle sending multiple messages to agent
            await ctx.output.print(msg.get_text_content())
    except Exception as e:  # noqa: BLE001
        await ctx.output.print(f"Error executing prompt: {e}")


list_prompts_cmd = Command(
    name="list-prompts",
    description="List available prompts",
    execute_func=list_prompts,
    help_text=(
        "Show all prompts available in the current configuration.\n"
        "Each prompt is shown with its name and description."
    ),
    category="prompts",
)

prompt_cmd = Command(
    name="prompt",
    description="Execute a prompt",
    execute_func=prompt_command,
    usage="<name> [arg1=value1] [arg2=value2]",
    help_text=(
        "Execute a named prompt with optional arguments.\n\n"
        "Arguments:\n"
        "  name: Name of the prompt to execute\n"
        "  argN=valueN: Optional arguments for the prompt\n\n"
        "Examples:\n"
        "  /prompt greet\n"
        "  /prompt analyze file=test.py\n"
        "  /prompt search query='python code'"
    ),
    category="prompts",
)
