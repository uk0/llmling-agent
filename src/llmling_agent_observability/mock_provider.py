"""Observability provider that collects calls for testing."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal, ParamSpec, TypeVar

from llmling_agent_observability.base_provider import ObservabilityProvider


P = ParamSpec("P")
R = TypeVar("R")


@dataclass
class MockCall:
    """Record of a provider call."""

    call_type: Literal["span", "wrap_agent", "wrap_tool", "wrap_action"]
    name: str
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class MockProvider(ObservabilityProvider):
    """Mock provider that collects calls for testing."""

    calls: list[MockCall] = field(default_factory=list)

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[None]:
        """Record span creation."""
        kwargs = {"attributes": attributes}
        self.calls.append(MockCall(call_type="span", name=name, kwargs=kwargs))
        yield

    def wrap_agent[T](self, kls: type[T], name: str) -> type[T]:
        """Record agent wrapper creation."""
        self.calls.append(MockCall(call_type="wrap_agent", name=name))
        return kls

    def wrap_tool(self, func: Callable, name: str, **kwargs: Any) -> Callable:
        """Record tool wrapper creation."""
        self.calls.append(MockCall(call_type="wrap_tool", name=name, kwargs=kwargs))
        return func

    def wrap_action(
        self,
        func: Callable[P, R],
        msg_template: str | None = None,
        *,
        span_name: str | None = None,
        **kwargs: Any,
    ) -> Callable[P, R]:
        """Record action wrapper creation."""
        kwargs = {"msg_template": msg_template, **kwargs}
        name = span_name or msg_template or func.__name__
        self.calls.append(MockCall(call_type="wrap_action", name=name, kwargs=kwargs))
        return func
