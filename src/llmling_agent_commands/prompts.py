"""Prompt slash commands."""

from __future__ import annotations

from typing import TYPE_CHECKING

from slashed import Command, CommandContext, CommandError

from llmling_agent_commands.completers import PromptCompleter


if TYPE_CHECKING:
    from llmling_agent.agent.context import AgentContext

PROMPT_HELP = """\
Show prompts from configured prompt hubs.

Usage examples:
  /prompt role.reviewer            # Use builtin prompt
  /prompt openlit:code_review     # Use specific provider
  /prompt langfuse:explain@v2     # Use specific version
  /prompt some_prompt[var=value]  # With variables
"""


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


async def prompt_command(
    ctx: CommandContext[AgentContext],
    args: list[str],
    kwargs: dict[str, str],
):
    """Show prompt content."""
    if not args:
        await ctx.output.print(
            "Usage: /prompt <[provider:]identifier[@version][?var=val]>"
        )
        return

    try:
        prompt = await ctx.context.prompt_manager.get(args[0])
        await ctx.output.print("\nPrompt Content:\n" + prompt)
    except Exception as e:
        msg = f"Error getting prompt: {e}"
        raise CommandError(msg) from e


async def list_prompts(
    ctx: CommandContext[AgentContext],
    args: list[str],
    kwargs: dict[str, str],
):
    """List available prompts from all providers."""
    prompts = await ctx.context.prompt_manager.list_prompts()
    output_lines = ["\nAvailable prompts:"]

    for provider, provider_prompts in prompts.items():
        if not provider_prompts:
            continue

        output_lines.append(f"\n{provider}:")
        sorted_prompts = sorted(provider_prompts)

        # For builtin prompts we can show their description
        if provider == "builtin":
            for prompt_name in sorted_prompts:
                prompt = ctx.context.definition.prompts.system_prompts[prompt_name]
                desc = f" - {prompt.type}"
                output_lines.append(f"  {prompt_name:<30}{desc}")
        else:
            # For other providers, just show names
            output_lines.extend(f"  {prompt_name}" for prompt_name in sorted_prompts)
    await ctx.output.print("\n".join(output_lines))


prompt_cmd = Command(
    name="prompt",
    description="Show prompt content",
    execute_func=prompt_command,
    help_text=PROMPT_HELP,
    category="prompts",
    completer=PromptCompleter(),
)


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
