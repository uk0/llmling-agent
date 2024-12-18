"""Meta-prompt command implementation."""

from __future__ import annotations

from llmling_agent.commands.base import Command, CommandContext
from llmling_agent.commands.exceptions import CommandError
from llmling_agent.prompts import DEFAULT_PROMPTS, PromptLibrary
from llmling_agent.prompts.models import PromptTemplate


async def meta_command(
    ctx: CommandContext,
    args: list[str],
    kwargs: dict[str, str],
) -> None:
    """Generate a prompt using meta-prompts."""
    if not args:
        await ctx.output.print(
            "Usage: /meta <goal> --category <name> [--max-length <n>] [--chain]"
        )
        return

    goal = args[0]
    max_length = int(kwargs.get("max_length", 0)) or None
    chain = kwargs.get("chain", "").lower() == "true"  # Default: False

    # Get the requested style
    categories = {
        "role": kwargs.get("role"),
        "style": kwargs.get("style"),
        "format": kwargs.get("format"),
        "tone": kwargs.get("tone"),
        "pattern": kwargs.get("pattern"),
        "audience": kwargs.get("audience"),
        "purpose": kwargs.get("purpose"),
    }

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
            return

        if chain:
            # Sequential application (multiple LLM calls)
            current_prompt = goal
            model = ctx.session._agent.model_name
            for prompt in prompts:
                current_prompt = await prompt.apply(
                    current_prompt, model=model, max_length=max_length
                )
            result = current_prompt
        else:
            # Combined application (single LLM call)
            styles = list(categories.values())

            # Get the combiner prompt
            combiner = library.get_meta_prompt("internal.combine")

            # Create new template with combined system prompts
            combined_system = (
                combiner.system
                + "\n\nAvailable Styles:\n"
                + "\n\n".join(
                    f"Style '{name}':\n{p.system}" for name, p in zip(styles, prompts)
                )
            )

            combined = PromptTemplate(
                description="Combined styles",
                system=combined_system,
                template=combiner.template,
                variables=combiner.variables,
            )

            # Generate combined prompt
            style_names = [s for s in styles if s is not None]
            result = await combined.apply(
                goal=goal,
                styles=", ".join(style_names),
                model=ctx.session._agent.model_name,
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
    help_text=(
        "Generate a prompt using different styles and formats.\n\n"
        "Categories:\n"
        "  --role     : Role-based styles (reviewer, teacher, etc.)\n"
        "  --style    : Writing styles (formal, concise, etc.)\n"
        "  --format   : Output formats (markdown, bullet_points, etc.)\n"
        "  --tone     : Tone modifiers (professional, casual, etc.)\n"
        "  --pattern  : Structure patterns (problem_solution, etc.)\n\n"
        "Options:\n"
        "  --chain    : Apply multiple styles sequentially\n"
        "  --max-length: Maximum length constraint\n\n"
        "Examples:\n"
        "  /meta 'Review code' --role reviewer --style concise\n"
        "  /meta 'Explain API' --style pirate --format markdown\n"
        "  /meta 'Review code' --role reviewer --style pirate --chain true"
    ),
    category="prompts",
)
