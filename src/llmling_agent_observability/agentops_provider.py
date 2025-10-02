"""AgentOps-based observability provider."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
import os
from typing import TYPE_CHECKING, Any, cast

import agentops
from agentops.sdk.decorators import operation

from llmling_agent_observability.base_provider import ObservabilityProvider


if TYPE_CHECKING:
    from collections.abc import Iterator

    from llmling_agent_config.observability import AgentOpsProviderConfig


class AgentOpsProvider(ObservabilityProvider):
    """AgentOps implementation of observability provider."""

    def __init__(self, config: AgentOpsProviderConfig):
        """Initialize AgentOps with configuration."""
        key = config.api_key.get_secret_value() if config.api_key else None
        key = key or os.getenv("AGENTOPS_API_KEY")
        if not key:
            msg = "No API key provided for AgentOps"
            raise RuntimeError(msg)
        self.config = {
            "api_key": key,
            "parent_key": config.parent_key,
            "endpoint": config.endpoint,
            "max_wait_time": config.max_wait_time,
            "max_queue_size": config.max_queue_size,
            "default_tags": config.tags,
            "instrument_llm_calls": config.instrument_llm_calls,
            "auto_start_session": config.auto_start_session,
            "inherited_session_id": config.inherited_session_id,
            "skip_auto_end_session": config.skip_auto_end_session,
        }
        # Remove None values to use AgentOps defaults
        self.config = {k: v for k, v in self.config.items() if v is not None}
        agentops.init(**self.config)

    def wrap_action[R, **P](
        self,
        func: Callable[P, R],
        msg_template: str | None = None,
        *,
        span_name: str | None = None,
    ) -> Callable[P, R]:
        """Wrap a function with AgentOps tracking."""
        name = span_name or msg_template or getattr(func, "__name__", "Unknown")
        wrapped = operation(name)(func)
        return cast(Callable[P, R], wrapped)

    def wrap_tool[T](self, func: Callable[..., T], name: str) -> Callable[..., T]:
        """Wrap a tool function with AgentOps tracking."""
        wrapped = operation(name)(func)
        return cast(Callable[..., T], wrapped)

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[Any]:
        """Create an AgentOps span."""
        # Since record_action isn't a context manager, we create our own
        event_wrapper = operation(name)

        # Create a dummy function for the event
        def span_func():
            pass

        wrapped = event_wrapper(span_func)
        try:
            wrapped()  # Start the event
            yield None
        finally:
            # End of context will automatically finish the event
            pass


if __name__ == "__main__":
    import asyncio

    import pydantic_ai  # noqa: F401

    from llmling_agent import Agent
    from llmling_agent.observability import registry
    from llmling_agent_config.observability import AgentOpsProviderConfig

    config = AgentOpsProviderConfig()
    provider = AgentOpsProvider(config)
    registry.configure_provider(provider)
    agent = Agent[None](model="openai:gpt-5-mini", name="test")

    @agent.tools.tool(name="test")
    def square(x: int) -> int:
        return x * x

    async def main():
        result = await agent.run("Square root of 16?")
        await asyncio.sleep(2)
        return result

    asyncio.run(main())
