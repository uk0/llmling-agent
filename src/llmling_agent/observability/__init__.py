"""Observability package."""

from __future__ import annotations

from llmling_agent.observability.observability_registry import ObservabilityRegistry

registry = ObservabilityRegistry()

__all__ = ["ObservabilityRegistry", "registry"]
