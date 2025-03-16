"""Observability provider using Arize Phoenix."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
import os
from typing import Any, ParamSpec, TypeVar

from arize.otel import Transport, register
from openinference.instrumentation import using_attributes
from openinference.instrumentation.litellm import LiteLLMInstrumentor
from openinference.instrumentation.openai import OpenAIInstrumentor

from llmling_agent_config.observability import ArizePhoenixProviderConfig
from llmling_agent_observability.base_provider import ObservabilityProvider


P = ParamSpec("P")
R = TypeVar("R")


class ArizePhoenixProvider(ObservabilityProvider):
    """Observability provider using Arize Phoenix."""

    def __init__(self, config: ArizePhoenixProviderConfig):
        self.config = config
        self._tracer_provider = None
        self._configure()

    def _configure(self):
        """Initialize Arize Phoenix with OpenTelemetry."""
        key = self.config.api_key.get_secret_value() if self.config.api_key else None
        key = key or os.getenv("ARIZE_API_KEY")
        if not key:
            msg = "No API key set for Arize."
            raise RuntimeError(msg)
        space = self.config.space_key or os.getenv("ARIZE_SPACE_ID") or "default"
        self._tracer_provider = register(
            space_id=space, api_key=key, transport=Transport.GRPC
        )

        # Instrument underlying LLM libraries
        LiteLLMInstrumentor().instrument(tracer_provider=self._tracer_provider)
        OpenAIInstrumentor().instrument(tracer_provider=self._tracer_provider)

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[None]:
        """Create a trace span using OpenTelemetry context."""
        with using_attributes(
            tags=[self.config.environment] if self.config.environment else None,
            metadata=attributes,
            session_id=self.config.space_key or "",
        ):
            yield

    def wrap_tool(self, func: Callable, name: str) -> Callable:
        """Wrap tool with OpenTelemetry context."""

        @using_attributes(
            tags=["tool", self.config.environment]
            if self.config.environment
            else ["tool"],
            metadata={"tool_name": name},
            session_id=self.config.space_key or "",
        )
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        return wrapped

    def wrap_action(
        self,
        func: Callable[P, R],
        msg_template: str | None = None,
        *,
        span_name: str | None = None,
    ) -> Callable[P, R]:
        """Wrap action with OpenTelemetry context."""
        name = span_name or msg_template or func.__name__

        @using_attributes(
            tags=["action", self.config.environment]
            if self.config.environment
            else ["action"],
            metadata={"action_name": name},
            session_id=self.config.space_key or "",
        )
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            return func(*args, **kwargs)

        return wrapped


if __name__ == "__main__":
    import asyncio

    from llmling_agent import Agent
    from llmling_agent_config.observability import ArizePhoenixProviderConfig

    config = ArizePhoenixProviderConfig(environment="dev")
    provider = ArizePhoenixProvider(config)

    agent = Agent[None](model="gpt-4o-mini", name="test")

    @agent.tools.tool(name="test")
    def square(x: int) -> int:
        return x * x

    with provider.span("test"):
        pass

    async def main():
        result = await agent.run("Square root of 16?")
        print(result)
        await asyncio.sleep(5)

    asyncio.run(main())
