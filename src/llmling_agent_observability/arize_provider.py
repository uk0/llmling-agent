from collections.abc import Callable, Iterator
from contextlib import contextmanager
import os
from typing import Any, ParamSpec, TypeVar

from openinference.instrumentation import using_attributes
from openinference.instrumentation.litellm import LiteLLMInstrumentor
from openinference.instrumentation.openai import OpenAIInstrumentor
from phoenix.otel import register

from llmling_agent.models.observability import ArizePhoenixProviderConfig
from llmling_agent_observability.base_provider import ObservabilityProvider


P = ParamSpec("P")
R = TypeVar("R")


class ArizePhoenixProvider(ObservabilityProvider):
    def __init__(self, config: ArizePhoenixProviderConfig):
        self.config = config
        self._tracer_provider = None
        self._configure()

    def _configure(self) -> None:
        """Initialize Arize Phoenix with OpenTelemetry."""
        if self.config.api_key:
            os.environ["PHOENIX_CLIENT_HEADERS"] = f"api_key={self.config.api_key}"

        os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "https://app.phoenix.arize.com"

        # Configure the Phoenix tracer with batch processing for production
        self._tracer_provider = register(
            project_name=self.config.space_key or "default",
            batch=True,  # Use batch processing in production
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
