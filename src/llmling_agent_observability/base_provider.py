"""Base classes for observability providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")


class ObservabilityProvider(ABC):
    """Base class for observability providers."""

    @abstractmethod
    def wrap_agent(self, func: Callable[..., T], name: str) -> Callable[..., T]:
        """Wrap an agent class with observability."""

    @abstractmethod
    def wrap_tool(self, func: Callable[..., T], name: str) -> Callable[..., T]:
        """Wrap a tool function with observability."""

    @abstractmethod
    def wrap_action(
        self,
        func: Callable[P, R],
        msg_template: str | None = None,
        *,
        span_name: str | None = None,
    ) -> Callable[P, R]:
        """Wrap a function with observability tracking."""

    @abstractmethod
    @contextmanager
    def span(self, name: str) -> Iterator[Any]:
        """Create a span for manual instrumentation."""
