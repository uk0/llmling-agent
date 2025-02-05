"""Base classes for observability providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from llmling_agent.models.observability import (
    AgentOpsProviderConfig,
    ArizePhoenixProviderConfig,
    LangsmithProviderConfig,
    LogfireProviderConfig,
)


if TYPE_CHECKING:
    from collections.abc import Callable

    from llmling_agent.models.observability import ObservabilityConfig
    from llmling_agent_observability.base_provider import ObservabilityProvider


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
    def register_providers(cls, observability_config: ObservabilityConfig) -> None:
        """Register and configure all observability providers.

        Args:
            observability_config: Configuration for observability providers
        """
        if not observability_config.enabled:
            return

        for library in observability_config.instrument_libraries or []:
            match library:
                case "pydantic_ai":
                    import pydantic_ai  # noqa
                case "litellm":
                    import litellm  # noqa

        # Configure each provider
        for provider_config in observability_config.providers:
            match provider_config:
                case LogfireProviderConfig():
                    from llmling_agent_observability.logfire_provider import (
                        LogfireProvider,
                    )

                    provider: ObservabilityProvider = LogfireProvider(
                        provider_config.token,
                        provider_config.service_name,
                        provider_config.environment,
                    )
                    cls.configure_provider(provider)
                case AgentOpsProviderConfig():
                    from llmling_agent_observability.agentops_provider import (
                        AgentOpsProvider,
                    )

                    provider = AgentOpsProvider(
                        provider_config.api_key, provider_config.tags
                    )
                    cls.configure_provider(provider)

                case LangsmithProviderConfig():
                    from llmling_agent_observability.langsmith_provider import (
                        LangsmithProvider,
                    )

                    provider = LangsmithProvider(provider_config)
                    cls.configure_provider(provider)
                case ArizePhoenixProviderConfig():
                    from llmling_agent_observability.arize_provider import (
                        ArizePhoenixProvider,
                    )

                    provider = ArizePhoenixProvider(provider_config)
                    cls.configure_provider(provider)

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
