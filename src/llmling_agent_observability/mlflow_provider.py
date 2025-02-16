"""Observability provider using MlFlow."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, LiteralString, ParamSpec, TypeVar

import mlflow
from mlflow.entities import SpanType

from llmling_agent_observability.base_provider import ObservabilityProvider


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from llmling_agent_config.observability import MlFlowProviderConfig


P = ParamSpec("P")
R = TypeVar("R")


class MlFlowProvider(ObservabilityProvider):
    """Observability provider using MlFlow."""

    def __init__(self, config: MlFlowProviderConfig):
        self.config = config
        if config.tracking_uri:
            mlflow.set_tracking_uri(config.tracking_uri)

    def wrap_tool[T](self, func: Callable[..., T], name: str) -> Callable[..., T]:
        """Wrap a tool function with MlFlow instrumentation."""
        return mlflow.trace(name=name, span_type=SpanType.TOOL)(func)

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[Any]:
        """Create a MlFlow span for manual instrumentation."""
        with mlflow.start_span(name, attributes=attributes) as span:
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
        return mlflow.trace(name=span_name)(func)
