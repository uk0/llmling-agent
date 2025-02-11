from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, LiteralString, ParamSpec, TypeVar

import braintrust

from llmling_agent_observability.base_provider import ObservabilityProvider


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from llmling_agent.models.observability import BraintrustProviderConfig


P = ParamSpec("P")
R = TypeVar("R")


class BraintrustProvider(ObservabilityProvider):
    def __init__(self, config: BraintrustProviderConfig):
        self.config = config
        key = self.config.api_key.get_secret_value() if self.config.api_key else None

        braintrust.init_logger(api_key=key)

    def wrap_tool[T](self, func: Callable[..., T], name: str) -> Callable[..., T]:
        """Wrap a tool function with braintrust instrumentation."""
        return braintrust.traced(name=name)(func)

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[Any]:
        """Create a braintrust span for manual instrumentation."""
        with braintrust.start_span(name, span_attributes=attributes) as span:
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
        return braintrust.traced(name=span_name)(func)
