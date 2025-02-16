"""Logfire-based observability provider."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, LiteralString, TypeVar

import logfire
from typing_extensions import ParamSpec

from llmling_agent.log import get_logger
from llmling_agent_observability.base_provider import ObservabilityProvider


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from llmling_agent_config.observability import LogfireProviderConfig


P = ParamSpec("P")
R = TypeVar("R")

logger = get_logger(__name__)


class LogfireProvider(ObservabilityProvider):
    """Logfire implementation of observability provider."""

    def __init__(self, config: LogfireProviderConfig, **kwargs: Any):
        """Initialize Logfire with configuration."""
        self.config = config
        token = self.config.token.get_secret_value() if self.config.token else None
        logfire.configure(
            token=token,
            service_name=config.service_name,
            environment=config.environment,
            **kwargs,
        )

    def wrap_tool[T](self, func: Callable[..., T], name: str) -> Callable[..., T]:
        """Wrap a tool function with logfire instrumentation."""
        return logfire.instrument(
            span_name=f"tool.{name}",
            extract_args=True,
        )(func)

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[Any]:
        """Create a logfire span for manual instrumentation."""
        with logfire.span(name, **attributes) as span:
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
