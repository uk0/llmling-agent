"""Command completion providers."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, get_args

from pydantic_ai.models import KnownModelName
from slashed import CompletionItem, CompletionProvider

from llmling_agent.prompts import DEFAULT_PROMPTS, PromptLibrary


if TYPE_CHECKING:
    from collections.abc import Iterator

    from slashed import CompletionContext

    from llmling_agent.chat_session.base import AgentChatSession


def get_tool_names(ctx: CompletionContext[AgentChatSession]) -> list[str]:
    """Get available tool names."""
    return list(ctx.command_context.context._agent.tools.keys())


def get_available_agents(ctx: CompletionContext[AgentChatSession]) -> list[str]:
    """Get available agent names."""
    if not ctx.command_context.context.pool:
        return []
    return ctx.command_context.context.pool.list_agents()


def get_resource_names(ctx: CompletionContext[AgentChatSession]) -> list[str]:
    """Get available resource names."""
    resources = ctx.command_context.context._agent.runtime.get_resources()
    return [r.name or "" for r in resources]


def get_prompt_names(ctx: CompletionContext[AgentChatSession]) -> list[str]:
    """Get available prompt names."""
    prompts = ctx.command_context.context._agent.runtime.get_prompts()
    return [p.name or "" for p in prompts]


def get_model_names(ctx: CompletionContext[AgentChatSession]) -> list[str]:
    """Get available model names from pydantic-ai and current configuration.

    Returns:
    - All models from KnownModelName literal type
    - Plus any additional models from current configuration
    """
    # Get models directly from the Literal type
    known_models = list(get_args(KnownModelName))

    agent = ctx.command_context.context._agent
    if not agent._context or not agent._context.definition:
        return known_models

    # Add any additional models from the current configuration
    agents = agent._context.definition.agents
    config_models = {str(a.model) for a in agents.values() if a.model is not None}

    # Combine both sources, keeping order (known models first)
    all_models = known_models[:]
    for model in config_models:
        if model not in all_models:
            all_models.append(model)

    return all_models


@lru_cache(maxsize=1)
def _load_prompt_library() -> PromptLibrary | None:
    """Load and cache the prompt library."""
    try:
        return PromptLibrary.from_file(DEFAULT_PROMPTS)
    except Exception:  # noqa: BLE001
        return None


class MetaCompleter(CompletionProvider):
    """Smart completer for meta-prompts."""

    def get_completions(
        self, ctx: CompletionContext[AgentChatSession]
    ) -> Iterator[CompletionItem]:
        """Complete meta command arguments."""
        current = ctx.current_word
        args = ctx.command_args

        # If completing a new argument that starts with --
        if current.startswith("--"):
            # Get categories that haven't been used yet
            used_categories = {
                arg[2:].split("=")[0] for arg in args if arg.startswith("--")
            }
            library = _load_prompt_library()
            if not library:
                return

            # Suggest unused categories
            categories = {
                name.split(".")[0] for name in library.meta_prompts if "." in name
            }
            for category in categories - used_categories:
                if category.startswith(current[2:]):
                    yield CompletionItem(
                        text=f"--{category}",
                        metadata="Meta prompt category",
                        kind="argument",  # type: ignore
                    )

        # If completing a value after a category
        for arg in args:
            if arg.startswith("--") and "=" not in arg:
                category = arg[2:]  # Remove --
                library = _load_prompt_library()
                if not library:
                    return

                # Get styles for this category
                styles = [
                    name.split(".")[1]
                    for name in library.meta_prompts
                    if name.startswith(f"{category}.")
                ]
                for style in styles:
                    if style.startswith(current):
                        yield CompletionItem(
                            text=style,
                            metadata=f"{category} style",
                            kind="value",  # type: ignore
                        )
