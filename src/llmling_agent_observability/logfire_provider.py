"""Logfire-based observability provider."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, LiteralString, TypeVar

import logfire
from typing_extensions import ParamSpec

from llmling_agent_observability.base_provider import ObservabilityProvider


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")


class LogfireProvider(ObservabilityProvider):
    """Logfire implementation of observability provider."""

    def __init__(
        self,
        token: str | None = None,
        service_name: str | None = None,
        environment: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize Logfire with configuration."""
        logfire.configure(
            token=token,
            service_name=service_name,
            environment=environment,
            **kwargs,
        )

    def wrap_agent(self, func: Callable[..., T], name: str) -> Callable[..., T]:
        """Wrap an agent class with logfire instrumentation."""
        return logfire.instrument(
            span_name=f"agent.{name}",
            extract_args=True,
        )(func)

    def wrap_tool(self, func: Callable[..., T], name: str) -> Callable[..., T]:
        """Wrap a tool function with logfire instrumentation."""
        return logfire.instrument(
            span_name=f"tool.{name}",
            extract_args=True,
        )(func)

    @contextmanager
    def span(self, name: str) -> Iterator[Any]:
        """Create a logfire span for manual instrumentation."""
        with logfire.span(name) as span:
            yield span

    def wrap_action(
        self,
        func: Callable[P, R],
        msg_template: LiteralString | None = None,
        *,
        span_name: str | None = None,
    ) -> Callable[P, R]:
        """Instrument a function with tracing.

        Args:
            func: The function to instrument
            msg_template: The message template for logging
            span_name: Optional span name for tracing
        """
        return logfire.instrument(msg_template, span_name=span_name)(func)
