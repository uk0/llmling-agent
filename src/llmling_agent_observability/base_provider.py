"""Base classes for observability providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import asynccontextmanager, contextmanager
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterator


P = ParamSpec("P")
R = TypeVar("R")


class ObservabilityProvider(ABC):
    """Base class for observability providers."""

    def wrap_agent[T](self, kls: type[T], name: str) -> type[T]:
        """Wrap an agent class with observability."""
        return kls

    @abstractmethod
    def wrap_tool[T](self, func: Callable[..., T], name: str) -> Callable[..., T]:
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
    def span(self, name: str, **attributes: Any) -> Iterator[Any]:
        """Create a span for manual instrumentation."""

    @asynccontextmanager
    async def aspan(self, name: str, **attributes: Any) -> AsyncIterator[Any]:
        """Create an async span for manual instrumentation."""
        with self.span(name, attributes=attributes) as span:
            yield span
