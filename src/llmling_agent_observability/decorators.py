"""Decorator interface for observability."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llmling_agent_observability.registry import registry


if TYPE_CHECKING:
    from collections.abc import Callable


def track_agent(name: str) -> Callable:
    """Register an agent class for observability tracking."""
    return registry.register("agent", name)


def track_tool(name: str) -> Callable:
    """Register a tool function for observability tracking."""
    return registry.register("tool", name)


def track_action(name: str) -> Callable:
    """Register an action function for observability tracking."""
    return registry.register("action", name)
