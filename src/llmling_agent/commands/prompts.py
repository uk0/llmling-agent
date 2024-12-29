"""Prompt-related commands."""

from __future__ import annotations

from typing import TYPE_CHECKING

from slashed import Command, CommandContext
from slashed.completers import CallbackCompleter

from llmling_agent.commands.completers import get_prompt_names


if TYPE_CHECKING:
    from llmling_agent.chat_session.base import AgentChatSession

EXECUTE_PROMPT_HELP = """\
Execute a named prompt with optional arguments.

Arguments:
  name: Name of the prompt to execute
  argN=valueN: Optional arguments for the prompt

Examples:
  /prompt greet
  /prompt analyze file=test.py
  /prompt search query='python code'
"""


async def list_prompts(
    ctx: CommandContext[AgentChatSession],
    args: list[str],
    kwargs: dict[str, str],
):
    """List available prompts."""
    prompts = ctx.context._agent.runtime.get_prompts()
    await ctx.output.print("\nAvailable prompts:")
    for prompt in prompts:
        await ctx.output.print(f"  {prompt.name:<20} - {prompt.description}")


async def prompt_command(
    ctx: CommandContext[AgentChatSession],
    args: list[str],
    kwargs: dict[str, str],
):
    """Execute a prompt.

    The first argument is the prompt name, remaining kwargs are prompt arguments.
    """
    if not args:
        await ctx.output.print("Usage: /prompt <name> [arg1=value1] [arg2=value2]")
        return

    name = args[0]
    try:
        await ctx.context._agent.conversation.add_context_from_prompt(
            name, arguments=kwargs
        )
        await ctx.output.print(f"Added prompt {name!r} to next message as context.")
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
    help_text=EXECUTE_PROMPT_HELP,
    category="prompts",
    completer=CallbackCompleter(get_prompt_names),
)
