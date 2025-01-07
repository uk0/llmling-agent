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

    from llmling_agent.chat_session.base import AgentPoolView


def get_tool_names(ctx: CompletionContext[AgentPoolView]) -> list[str]:
    """Get available tool names."""
    return list(ctx.command_context.context._agent.tools.keys())


def get_available_agents(ctx: CompletionContext[AgentPoolView]) -> list[str]:
    """Get available agent names."""
    if not ctx.command_context.context.pool:
        return []
    return ctx.command_context.context.pool.list_agents()


def get_resource_names(ctx: CompletionContext[AgentPoolView]) -> list[str]:
    """Get available resource names."""
    resources = ctx.command_context.context._agent.runtime.get_resources()
    return [r.name or "" for r in resources]


def get_prompt_names(ctx: CompletionContext[AgentPoolView]) -> list[str]:
    """Get available prompt names."""
    prompts = ctx.command_context.context._agent.runtime.get_prompts()
    return [p.name or "" for p in prompts]


def get_model_names(ctx: CompletionContext[AgentPoolView]) -> list[str]:
    """Get available model names from pydantic-ai and current configuration.

    Returns:
    - All models from KnownModelName literal type
    - Plus any additional models from current configuration
    """
    # Get models directly from the Literal type
    known_models = list(get_args(KnownModelName))

    agent = ctx.command_context.context._agent
    if not agent.context.definition:
        return known_models

    # Add any additional models from the current configuration
    agents = agent.context.definition.agents
    config_models = {str(a.model) for a in agents.values() if a.model is not None}

    # Combine both sources, keeping order (known models first)
    all_models = known_models[:]
    all_models.extend(model for model in config_models if model not in all_models)

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
        self, ctx: CompletionContext[AgentPoolView]
    ) -> Iterator[CompletionItem]:
        """Complete meta command arguments."""
        current = ctx.current_word
        args = ctx.command_args
        lib = _load_prompt_library()
        if not lib:
            return
        prompts = lib.meta_prompts
        # If completing a new argument that starts with --
        if current.startswith("--"):
            # Get categories that haven't been used yet
            used_cats = {arg[2:].split("=")[0] for arg in args if arg.startswith("--")}
            # Suggest unused categories
            cats = {name.split(".")[0] for name in prompts if "." in name}
            for category in cats - used_cats:
                if category.startswith(current[2:]):
                    meta = "Meta prompt category"
                    text = f"--{category}"
                    yield CompletionItem(text=text, metadata=meta, kind="argument")  # type: ignore
        # If completing a value after a category
        for arg in args:
            if not (arg.startswith("--") and "=" not in arg):
                continue
            category = arg[2:]  # Remove --
            # Get styles for this category
            styles = [n.split(".")[1] for n in prompts if n.startswith(f"{category}.")]
            for style in styles:
                if style.startswith(current):
                    meta = f"{category} style"
                    yield CompletionItem(text=style, metadata=meta, kind="value")  # type: ignore
