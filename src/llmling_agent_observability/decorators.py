from __future__ import annotations

from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

from llmling_agent.log import get_logger
from llmling_agent_observability.registry import registry


logger = get_logger(__name__)

P = ParamSpec("P")
R = TypeVar("R")
F = TypeVar("F", bound=Callable[..., Any])
T = TypeVar("T", bound=type)


def track_agent(name: str | None = None, **kwargs: Any) -> Callable[[T], T]:
    """Register a class for agent tracking."""

    def decorator(cls: T) -> T:
        agent_name = name or cls.__name__
        logger.debug("Registering agent class %r as %r", cls.__name__, agent_name)
        registry.register_agent(agent_name, cls, **kwargs)
        return cls

    return decorator


def track_tool(name: str | None = None, **kwargs: Any) -> Callable[[F], F]:
    """Register a function for tool tracking."""

    def decorator(func: F) -> F:
        tool_name = name or func.__name__
        logger.debug("Registering tool function %r as %r", func.__name__, tool_name)
        registry.register_tool(tool_name, func, **kwargs)
        return func

    return decorator


def track_action(msg_template: str | None = None, **kwargs: Any) -> Callable[[F], F]:
    """Register a function for action tracking."""

    def decorator(func: F) -> F:
        action_name = msg_template or func.__name__
        logger.debug(
            "Registering action %r with template %r and args %r",
            func.__name__,
            action_name,
            kwargs,
        )
        registry.register_action(action_name, func, **kwargs)
        return func

    return decorator
