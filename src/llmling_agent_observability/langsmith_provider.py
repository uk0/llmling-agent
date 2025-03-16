"""Observability provider using Langsmith."""

from __future__ import annotations

from contextlib import contextmanager
import os
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

from langsmith import Client
from langsmith.run_helpers import traceable

from llmling_agent_observability.base_provider import ObservabilityProvider


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from llmling_agent_config.observability import LangsmithProviderConfig

P = ParamSpec("P")
R = TypeVar("R")


class LangsmithProvider(ObservabilityProvider):
    """Observability provider using Langsmith."""

    def __init__(self, config: LangsmithProviderConfig):
        self.config = config
        self._client: Client | None = None
        self._configure()

    def _configure(self):
        """Set up Langsmith client and environment."""
        if self.config.project_name:
            os.environ["LANGCHAIN_PROJECT"] = self.config.project_name
        if self.config.environment:
            os.environ["LANGCHAIN_ENVIRONMENT"] = self.config.environment
        key = self.config.api_key.get_secret_value() if self.config.api_key else None

        self._client = Client(api_key=key)

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[None]:
        """Create a trace span using Langsmith RunTree."""
        if self._client is None:
            yield
            return

        # Create run with required parameters
        self._client.create_run(
            name=name,
            inputs=attributes or {},
            run_type="chain",  # or could be parameterized
            project_name=self.config.project_name,
            tags=self.config.tags,
        )
        try:
            yield
        finally:
            # We might want to update the run with results here
            # but that would require capturing the yielded context somehow
            pass

    def wrap_tool(self, func: Callable, name: str) -> Callable:
        return traceable(
            run_type="tool",
            name=name,
            tags=self.config.tags,
        )(func)

    def wrap_action(
        self,
        func: Callable[P, R],
        msg_template: str | None = None,
        *,
        span_name: str | None = None,
    ) -> Callable[P, R]:
        """Wrap action with Langsmith tracing."""
        name = span_name or msg_template or func.__name__
        return traceable(  # type: ignore
            run_type="llm",
            name=name,
            tags=[*self.config.tags, "action"],
        )(func)
