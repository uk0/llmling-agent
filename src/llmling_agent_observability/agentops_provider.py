"""AgentOps-based observability provider."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, TypeVar, cast

import agentops

from llmling_agent_observability.base_provider import ObservabilityProvider


if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import ParamSpec

    P = ParamSpec("P")
    R = TypeVar("R")

T = TypeVar("T")


class AgentOpsProvider(ObservabilityProvider):
    """AgentOps implementation of observability provider."""

    def __init__(
        self,
        api_key: str | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize AgentOps with configuration."""
        self.config = {"api_key": api_key, "tags": tags, **kwargs}
        agentops.init(**self.config)

    def wrap_action(
        self,
        func: Callable[P, R],
        msg_template: str | None = None,
        *,
        span_name: str | None = None,
    ) -> Callable[P, R]:
        """Wrap a function with AgentOps tracking."""
        name = span_name or msg_template or func.__name__
        wrapped = agentops.record_action(name)(func)
        return cast(Callable[P, R], wrapped)

    def wrap_agent(self, func: Callable[..., T], name: str) -> Callable[..., T]:
        """Wrap an agent class with AgentOps tracking."""
        if not isinstance(func, type):
            msg = "AgentOps @track_agent can only be used with classes"
            raise TypeError(msg)
        wrapped = agentops.track_agent(name)(func)
        return cast(Callable[..., T], wrapped)

    def wrap_tool(self, func: Callable[..., T], name: str) -> Callable[..., T]:
        """Wrap a tool function with AgentOps tracking."""
        wrapped = agentops.record_tool(name)(func)
        return cast(Callable[..., T], wrapped)

    @contextmanager
    def span(self, name: str) -> Iterator[Any]:
        """Create an AgentOps span."""
        # Since record_action isn't a context manager, we create our own
        event_wrapper = agentops.record_action(name)

        # Create a dummy function for the event
        def span_func() -> None:
            pass

        wrapped = event_wrapper(span_func)
        try:
            wrapped()  # Start the event
            yield None
        finally:
            # End of context will automatically finish the event
            pass
