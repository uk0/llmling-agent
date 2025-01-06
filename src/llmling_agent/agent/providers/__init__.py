"""Provider implementations for different agent types."""

from __future__ import annotations

from typing import TypeVar

from llmling_agent.agent.providers.base import (
    AgentProvider,
    ProviderResponse,
)
from llmling_agent.agent.providers.human import HumanProvider
from llmling_agent.agent.providers.pydanticai import PydanticAIProvider

TDeps = TypeVar("TDeps")

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
