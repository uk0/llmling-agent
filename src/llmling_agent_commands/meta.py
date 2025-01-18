"""Meta-prompt command implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from slashed import Command, CommandContext, CommandError

from llmling_agent.prompts import DEFAULT_PROMPTS, PromptLibrary
from llmling_agent.prompts.models import PromptTemplate
from llmling_agent_commands.completers import MetaCompleter


if TYPE_CHECKING:
    from llmling_agent.chat_session.base import AgentPoolView


META_HELP = """\
Generate a prompt using different styles and formats.

Categories:
  --role     : Role-based styles (reviewer, teacher, etc.)
  --style    : Writing styles (formal, concise, etc.)
  --format   : Output formats (markdown, bullet_points, etc.)
  --tone     : Tone modifiers (professional, casual, etc.)
  --pattern  : Structure patterns (problem_solution, etc.)

Options:
  --chain    : Apply multiple styles sequentially
  --max-length: Maximum length constraint

Examples:
  /meta 'Review code' --role reviewer --style concise
  /meta 'Explain API' --style pirate --format markdown
  /meta 'Review code' --role reviewer --style pirate --chain true
"""


async def meta_command(
    ctx: CommandContext[AgentPoolView],
    args: list[str],
    kwargs: dict[str, str],
):
    """Generate a prompt using meta-prompts."""
    if not args:
        msg = "Usage: /meta <goal> --category <name> [--max-length <n>] [--chain]"
        await ctx.output.print(msg)
        return

    goal = args[0]
    max_length = int(kwargs.get("max_length", 0)) or None
    chain = kwargs.get("chain", "").lower() == "true"  # Default: False
    styles = ["role", "style", "format", "tone", "pattern", "audiene", "purpose"]
    categories = {style: kwargs.get(style) for style in styles}
    # Filter out None values
    categories = {k: v for k, v in categories.items() if v is not None}

    try:
        library = PromptLibrary.from_file(DEFAULT_PROMPTS)
        prompts = []
        for category, name in categories.items():
            template_name = f"{category}.{name}"
            try:
                prompt = library.get_meta_prompt(template_name)
                prompts.append(prompt)
            except KeyError as e:
                msg = f"Meta prompt not found: {template_name}"
                raise CommandError(msg) from e

        if not prompts:
            await ctx.output.print("No meta prompt selected via kwargs")
            return

        if chain:
            # Sequential application (multiple LLM calls)
            current_prompt = goal
            model = ctx.context._agent.model_name
            for prompt in prompts:
                current_prompt = await prompt.apply(
                    current_prompt,
                    model=model,
                    max_length=max_length,
                )
            result = current_prompt
        else:
            # Combined application (single LLM call)
            # Get the combiner prompt
            combiner = library.get_meta_prompt("internal.combine")
            zipped = zip(categories.values(), prompts)
            labels = [f"Style '{name}':\n{p.system}" for name, p in zipped]
            # Create new template with combined system prompts
            sys_prompt = combiner.system + "\n\nAvailable Styles:\n" + "\n\n".join(labels)
            combined = PromptTemplate(
                description="Combined styles",
                system=sys_prompt,
                template=combiner.template,
                variables=combiner.variables,
            )

            # Generate combined prompt
            style_names = [s for s in styles if s is not None]
            result = await combined.apply(
                goal=goal,
                styles=", ".join(style_names),
                model=ctx.context._agent.model_name,  # type: ignore
                max_length=max_length,
            )
        await ctx.output.print("\nGenerated Prompt:\n" + result)

    except Exception as e:
        msg = f"Error generating prompt: {e}"
        raise CommandError(msg) from e


meta_cmd = Command(
    name="meta",
    description="Generate a prompt using meta-prompts",
    execute_func=meta_command,
    usage=(
        "<goal> [--role <name>] [--style <name>] [--format <name>] "
        "[--tone <name>] [--pattern <name>] [--max-length <n>] [--chain]"
    ),
    help_text=META_HELP,
    category="prompts",
    completer=MetaCompleter(),
)
