"""Observability package."""

from __future__ import annotations

from llmling_agent.observability.decorators import track_agent, track_action, track_tool
from llmling_agent.observability.observability_registry import ObservabilityRegistry

registry = ObservabilityRegistry()

__all__ = [
    "ObservabilityRegistry",
    "registry",
    "track_action",
    "track_agent",
    "track_tool",
]
