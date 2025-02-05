"""Base classes for observability providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, ParamSpec, TypeVar


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


@dataclass
class PendingDecoration:
    """Stores information about a pending decoration."""

    func: Callable
    name: str
    type: str  # 'agent', 'tool', 'action'


class ObservabilityRegistry:
    """Registry for pending decorations and provider configuration."""

    _pending: ClassVar[list[PendingDecoration]] = []
    _provider: ObservabilityProvider | None = None

    @classmethod
    def register(cls, type_: str, name: str) -> Callable:
        """Register a function for later decoration."""

        def decorator(func: Callable) -> Callable:
            cls._pending.append(PendingDecoration(func, name, type_))
            return func

        return decorator

    @classmethod
    def configure_provider(cls, provider: ObservabilityProvider) -> None:
        """Configure the provider and apply pending decorations."""
        cls._provider = provider
        for pending in cls._pending:
            match pending.type:
                case "agent":
                    pending.func = provider.wrap_agent(pending.func, pending.name)
                case "tool":
                    pending.func = provider.wrap_tool(pending.func, pending.name)
                case "action":
                    pending.func = provider.wrap_action(pending.func, pending.name)

    @classmethod
    def get_provider(cls) -> ObservabilityProvider | None:
        """Get the configured provider."""
        return cls._provider


registry = ObservabilityRegistry()
