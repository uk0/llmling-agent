"""Configuration models for observability providers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr


if TYPE_CHECKING:
    from llmling_agent_observability.base_provider import ObservabilityProvider


class BaseObservabilityProviderConfig(BaseModel):
    """Base configuration for observability providers."""

    type: str = Field(init=True)
    """Observability provider."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True, extra="forbid")

    def get_provider(self) -> ObservabilityProvider:
        """Get the provider instance."""
        raise NotImplementedError


class LogfireProviderConfig(BaseObservabilityProviderConfig):
    """Configuration for Logfire provider."""

    type: Literal["logfire"] = Field("logfire", init=False)

    token: SecretStr | None = None
    """Logfire API token."""

    service_name: str | None = None
    """Service name for tracing."""

    environment: str | None = None
    """Environment name (dev/prod/etc)."""

    def get_provider(self) -> ObservabilityProvider:
        """Get the provider instance."""
        from llmling_agent_observability.logfire_provider import LogfireProvider

        return LogfireProvider(self)


class AgentOpsProviderConfig(BaseObservabilityProviderConfig):
    """Configuration for AgentOps provider."""

    type: Literal["agentops"] = Field("agentops", init=False)

    api_key: SecretStr | None = None
    """AgentOps API key."""

    parent_key: str | None = None
    """Parent key for session inheritance."""

    endpoint: str | None = None
    """Custom endpoint URL for AgentOps service."""

    max_wait_time: int | None = None
    """Maximum time to wait for batch processing in seconds."""

    max_queue_size: int | None = None
    """Maximum size of the event queue."""

    tags: list[str] | None = None
    """Tags to apply to all events in this session."""

    instrument_llm_calls: bool | None = None
    """Whether to automatically instrument LLM API calls."""

    auto_start_session: bool | None = None
    """Whether to automatically start a session on initialization."""

    inherited_session_id: str | None = None
    """Session ID to inherit from."""

    skip_auto_end_session: bool | None = None
    """Whether to skip auto-ending the session on exit."""

    def get_provider(self) -> ObservabilityProvider:
        """Get the provider instance."""
        from llmling_agent_observability.agentops_provider import AgentOpsProvider

        return AgentOpsProvider(self)


class LangsmithProviderConfig(BaseObservabilityProviderConfig):
    """Configuration for Langsmith provider."""

    type: Literal["langsmith"] = Field("langsmith", init=False)

    api_key: SecretStr | None = None
    """Langsmith API key."""

    project_name: str | None = None
    """Project name in Langsmith."""

    tags: list[str] = Field(default_factory=list)
    """Tags to apply to traces."""

    environment: str | None = None
    """Environment name (dev/prod/staging)."""

    def get_provider(self) -> ObservabilityProvider:
        """Get the provider instance."""
        from llmling_agent_observability.langsmith_provider import LangsmithProvider

        return LangsmithProvider(self)


class ArizePhoenixProviderConfig(BaseObservabilityProviderConfig):
    """Configuration for Arize Phoenix provider."""

    type: Literal["arize"] = Field("arize", init=False)

    api_key: SecretStr | None = None
    """Arize API key."""

    space_key: str | None = None
    """Arize workspace identifier."""

    model_id: str | None = None
    """Model identifier in Arize."""

    environment: str | None = None
    """Environment name."""

    def get_provider(self) -> ObservabilityProvider:
        """Get the provider instance."""
        from llmling_agent_observability.arize_provider import ArizePhoenixProvider

        return ArizePhoenixProvider(self)


class MlFlowProviderConfig(BaseObservabilityProviderConfig):
    """Configuration for MlFlow provider."""

    type: Literal["mlflow"] = Field("mlflow", init=False)

    tracking_uri: str | None = None
    """Tracking URI for MLFlow."""

    def get_provider(self) -> ObservabilityProvider:
        """Get the provider instance."""
        from llmling_agent_observability.mlflow_provider import MlFlowProvider

        return MlFlowProvider(self)


class BraintrustProviderConfig(BaseObservabilityProviderConfig):
    """Configuration for Braintrust provider."""

    type: Literal["braintrust"] = Field("braintrust", init=False)

    api_key: SecretStr | None = None
    """Braintrust API key."""

    def get_provider(self) -> ObservabilityProvider:
        """Get the provider instance."""
        from llmling_agent_observability.braintrust_provider import BraintrustProvider

        return BraintrustProvider(self)


class LaminarProviderConfig(BaseObservabilityProviderConfig):
    """Configuration for Laminar provider."""

    type: Literal["laminar"] = Field("laminar", init=False)

    api_key: SecretStr | None = None
    """Laminar API key."""

    def get_provider(self) -> ObservabilityProvider:
        """Get the provider instance."""
        from llmling_agent_observability.laminar_provider import LaminarProvider

        return LaminarProvider(self)


class TraceloopProviderConfig(BaseObservabilityProviderConfig):
    """Configuration for Traceloop provider."""

    type: Literal["traceloop"] = Field("traceloop", init=False)

    api_key: SecretStr | None = None
    """Traceloop API key."""

    def get_provider(self) -> ObservabilityProvider:
        """Get the provider instance."""
        from llmling_agent_observability.traceloop_provider import TraceloopProvider

        return TraceloopProvider(self)


ObservabilityProviderConfig = Annotated[
    LogfireProviderConfig
    | AgentOpsProviderConfig
    | LangsmithProviderConfig
    | MlFlowProviderConfig
    | BraintrustProviderConfig
    | TraceloopProviderConfig
    | ArizePhoenixProviderConfig,
    Field(discriminator="type"),
]


class ObservabilityConfig(BaseModel):
    """Global observability configuration."""

    enabled: bool = True
    """Whether observability is enabled."""

    providers: list[ObservabilityProviderConfig] = Field(default_factory=list)
    """Provider-specific configuration."""

    instrument_libraries: list[str] | None = None
    """Which providers should be imported before initialization.

    By default, all libraries used in the AgentsManifest are instrumented.
    """

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")
