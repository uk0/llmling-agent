"""Observability provider using Laminar."""

from __future__ import annotations

from contextlib import contextmanager
import os
from typing import TYPE_CHECKING, Any, LiteralString, ParamSpec, TypeVar

from lmnr import Laminar, observe

from llmling_agent_observability.base_provider import ObservabilityProvider


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from llmling_agent_config.observability import LaminarProviderConfig


P = ParamSpec("P")
R = TypeVar("R")


class LaminarProvider(ObservabilityProvider):
    """Observability provider using Laminar."""

    def __init__(self, config: LaminarProviderConfig):
        self.config = config
        api_key = (
            self.config.api_key.get_secret_value()
            if self.config.api_key
            else os.environ["LMNR_PROJECT_API_KEY"]
        )
        Laminar.initialize(project_api_key=api_key)

    def wrap_tool[T](self, func: Callable[..., T], name: str) -> Callable[..., T]:
        """Wrap a tool function with Laminar instrumentation."""
        return observe(name=name)(func)

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[Any]:
        """Create a Laminar span for manual instrumentation."""
        with Laminar.start_as_current_span(name="my_custom_llm_call") as span:
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
        return observe(name=span_name)(func)
