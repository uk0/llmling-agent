"""AgentOps-based observability provider."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
import os
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar, cast

import agentops

from llmling_agent_observability.base_provider import ObservabilityProvider


if TYPE_CHECKING:
    from collections.abc import Iterator

    from llmling_agent.models.observability import AgentOpsProviderConfig

P = ParamSpec("P")
R = TypeVar("R")


class AgentOpsProvider(ObservabilityProvider):
    """AgentOps implementation of observability provider."""

    def __init__(self, config: AgentOpsProviderConfig):
        """Initialize AgentOps with configuration."""
        key = config.api_key.get_secret_value() if config.api_key else None
        key = key or os.getenv("AGENTOPS_API_KEY")
        if not key:
            msg = "No API key provided for AgentOps"
            raise RuntimeError(msg)
        self.config = {
            "api_key": key,
            "parent_key": config.parent_key,
            "endpoint": config.endpoint,
            "max_wait_time": config.max_wait_time,
            "max_queue_size": config.max_queue_size,
            "default_tags": config.tags,
            "instrument_llm_calls": config.instrument_llm_calls,
            "auto_start_session": config.auto_start_session,
            "inherited_session_id": config.inherited_session_id,
            "skip_auto_end_session": config.skip_auto_end_session,
        }
        # Remove None values to use AgentOps defaults
        self.config = {k: v for k, v in self.config.items() if v is not None}
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

    def wrap_agent[T](self, kls: type[T], name: str) -> type[T]:
        """Wrap an agent class with AgentOps tracking."""
        if not isinstance(kls, type):
            msg = "AgentOps @track_agent can only be used with classes"
            raise TypeError(msg)
        # Only pass the name to AgentOps
        wrapped = agentops.track_agent(name)(kls)
        return cast(type[T], wrapped)

    def wrap_tool[T](self, func: Callable[..., T], name: str) -> Callable[..., T]:
        """Wrap a tool function with AgentOps tracking."""
        wrapped = agentops.record_tool(name)(func)
        return cast(Callable[..., T], wrapped)

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[Any]:
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
