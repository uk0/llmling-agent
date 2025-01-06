"""Provider implementations for different agent types."""

from __future__ import annotations

from llmling_agent.agent.providers.base import AgentProvider, ProviderResponse
from llmling_agent.agent.providers.human import HumanProvider
from llmling_agent.agent.providers.pydanticai import PydanticAIProvider

type AnyProvider = PydanticAIProvider | HumanProvider

__all__ = [
    # Base types
    "AgentProvider",
    "AnyProvider",
    # Provider implementations
    "HumanProvider",
    "ProviderResponse",
    "PydanticAIProvider",
]
