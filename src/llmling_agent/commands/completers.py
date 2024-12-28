"""Command completion providers."""

from __future__ import annotations

from typing import TYPE_CHECKING, get_args

from pydantic_ai.models import KnownModelName


if TYPE_CHECKING:
    from slashed import CompletionContext

    from llmling_agent.chat_session.base import AgentChatSession


def get_tool_names(ctx: CompletionContext[AgentChatSession]) -> list[str]:
    """Get available tool names."""
    assert ctx.command_context
    return list(ctx.command_context.data._agent.tools.keys())


def get_available_agents(ctx: CompletionContext[AgentChatSession]) -> list[str]:
    """Get available agent names."""
    assert ctx.command_context
    if not ctx.command_context.data.pool:
        return []
    return ctx.command_context.data.pool.list_agents()


def get_resource_names(ctx: CompletionContext[AgentChatSession]) -> list[str]:
    """Get available resource names."""
    assert ctx.command_context
    resources = ctx.command_context.data._agent.runtime.get_resources()
    return [r.name or "" for r in resources]


def get_prompt_names(ctx: CompletionContext[AgentChatSession]) -> list[str]:
    """Get available prompt names."""
    assert ctx.command_context
    prompts = ctx.command_context.data._agent.runtime.get_prompts()
    return [p.name or "" for p in prompts]


def get_model_names(ctx: CompletionContext[AgentChatSession]) -> list[str]:
    """Get available model names from pydantic-ai and current configuration.

    Returns:
    - All models from KnownModelName literal type
    - Plus any additional models from current configuration
    """
    # Get models directly from the Literal type
    known_models = list(get_args(KnownModelName))

    assert ctx.command_context
    agent = ctx.command_context.data._agent
    if not agent._context or not agent._context.definition:
        return known_models

    # Add any additional models from the current configuration
    config_models = {
        str(a.model)
        for a in agent._context.definition.agents.values()
        if a.model is not None
    }

    # Combine both sources, keeping order (known models first)
    all_models = known_models[:]
    for model in config_models:
        if model not in all_models:
            all_models.append(model)

    return all_models
