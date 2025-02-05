from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from llmling_agent.log import get_logger
from llmling_agent.models.observability import (
    AgentOpsProviderConfig,
    ArizePhoenixProviderConfig,
    LangsmithProviderConfig,
    LogfireProviderConfig,
    ObservabilityConfig,
)


if TYPE_CHECKING:
    from collections.abc import Callable

    from llmling_agent_observability.base_provider import ObservabilityProvider

logger = get_logger(__name__)


@dataclass
class PendingDecoration:
    """Stores information about a pending decoration."""

    target: Any  # class or function to be decorated
    name: str
    kwargs: dict[str, Any]


class ObservabilityRegistry:
    """Registry for pending decorations and provider configuration."""

    def __init__(self) -> None:
        self._pending_agents: dict[str, PendingDecoration] = {}
        self._pending_tools: dict[str, PendingDecoration] = {}
        self._pending_actions: dict[str, PendingDecoration] = {}
        self._provider: ObservabilityProvider | None = None

    def register_agent(
        self,
        name: str,
        target: type[Any],
        **kwargs: Any,
    ) -> None:
        """Register a class for agent tracking."""
        self._pending_agents[name] = PendingDecoration(
            target=target,
            name=name,
            kwargs=kwargs,
        )
        logger.debug("Registered agent %r for observability tracking", name)

    def register_tool(
        self,
        name: str,
        target: Callable,
        **kwargs: Any,
    ) -> None:
        """Register a function for tool tracking."""
        self._pending_tools[name] = PendingDecoration(
            target=target,
            name=name,
            kwargs=kwargs,
        )
        logger.debug("Registered tool %r for observability tracking", name)

    def register_action(
        self,
        name: str,
        target: Callable,
        **kwargs: Any,
    ) -> None:
        """Register a function for action tracking."""
        self._pending_actions[name] = PendingDecoration(
            target=target,
            name=name,
            kwargs=kwargs,
        )
        msg = "Registered action %r for observability tracking with args %r"
        logger.debug(msg, name, kwargs)

    def configure_provider(self, provider: ObservabilityProvider) -> None:
        """Configure the provider and apply pending decorations."""
        logger.info(
            "Configuring observability provider: %s",
            provider.__class__.__name__,
        )
        self._provider = provider

        # Apply decorations for each type
        for pending in self._pending_agents.values():
            try:
                pending.target = provider.wrap_agent(
                    pending.target,
                    pending.name,
                    **pending.kwargs,
                )
                logger.debug("Applied agent tracking to %r", pending.name)
            except Exception:
                msg = "Failed to apply agent tracking to %r"
                logger.exception(msg, pending.name)

        for pending in self._pending_tools.values():
            try:
                pending.target = provider.wrap_tool(
                    pending.target,
                    pending.name,
                    **pending.kwargs,
                )
                logger.debug("Applied tool tracking to %r", pending.name)
            except Exception:
                msg = "Failed to apply tool tracking to %r"
                logger.exception(msg, pending.name)

        for pending in self._pending_actions.values():
            try:
                pending.target = provider.wrap_action(
                    pending.target,
                    msg_template=pending.name,
                    **pending.kwargs,
                )
                msg = "Applied action tracking to %r with args %r"
                logger.debug(msg, pending.name, pending.kwargs)
            except Exception:
                msg = "Failed to apply action tracking to %r"
                logger.exception(msg, pending.name)

    def register_providers(self, observability_config: ObservabilityConfig) -> None:
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
                    self.configure_provider(provider)
                case AgentOpsProviderConfig():
                    from llmling_agent_observability.agentops_provider import (
                        AgentOpsProvider,
                    )

                    provider = AgentOpsProvider(
                        provider_config.api_key, provider_config.tags
                    )
                    self.configure_provider(provider)

                case LangsmithProviderConfig():
                    from llmling_agent_observability.langsmith_provider import (
                        LangsmithProvider,
                    )

                    provider = LangsmithProvider(provider_config)
                    self.configure_provider(provider)
                case ArizePhoenixProviderConfig():
                    from llmling_agent_observability.arize_provider import (
                        ArizePhoenixProvider,
                    )

                    provider = ArizePhoenixProvider(provider_config)
                    self.configure_provider(provider)

    @property
    def provider(self) -> ObservabilityProvider | None:
        """Get the configured provider."""
        return self._provider


registry = ObservabilityRegistry()
