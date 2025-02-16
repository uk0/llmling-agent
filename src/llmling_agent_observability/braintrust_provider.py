"""Observability provider using Braintrust."""

from __future__ import annotations

from contextlib import contextmanager
import os
from typing import TYPE_CHECKING, Any, LiteralString, ParamSpec, TypeVar

import braintrust

from llmling_agent_observability.base_provider import ObservabilityProvider


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from llmling_agent_config.observability import BraintrustProviderConfig


P = ParamSpec("P")
R = TypeVar("R")


class BraintrustProvider(ObservabilityProvider):
    """Observability provider using Braintrust."""

    def __init__(self, config: BraintrustProviderConfig):
        self.config = config
        key = self.config.api_key.get_secret_value() if self.config.api_key else None
        key = key or os.environ.get("BRAINTRUST_API_KEY")
        if not key:
            msg = "Braintrust API key not found"
            raise ValueError(msg)
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
        return braintrust.traced(span_name)(func)


if __name__ == "__main__":
    import asyncio

    from llmling_agent import Agent
    from llmling_agent.observability import registry
    from llmling_agent_config.observability import BraintrustProviderConfig

    config = BraintrustProviderConfig()
    provider = BraintrustProvider(config)
    registry.configure_provider(provider)
    agent = Agent[None](model="gpt-4o-mini", name="test")

    @agent.tools.tool(name="test")
    def square(x: int) -> int:
        return x * x

    async def main():
        result = await agent.run("Square root of 16?")
        await asyncio.sleep(2)
        return result

    asyncio.run(main())
