"""Configuration models for observability providers."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class BaseObservabilityProviderConfig(BaseModel):
    """Base configuration for observability providers."""

    type: str = Field(init=False)
    """Observability provider."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True, extra="forbid")


class LogfireProviderConfig(BaseObservabilityProviderConfig):
    """Configuration for Logfire provider."""

    type: Literal["logfire"] = Field("logfire", init=False)

    token: str | None = None
    """Logfire API token."""

    service_name: str | None = None
    """Service name for tracing."""

    environment: str | None = None
    """Environment name (dev/prod/etc)."""


class AgentOpsProviderConfig(BaseObservabilityProviderConfig):
    """Configuration for AgentOps provider."""

    type: Literal["agentops"] = Field("agentops", init=False)

    api_key: str | None = None
    """AgentOps API key."""

    tags: list[str] = Field(default_factory=list)
    """Tags for session grouping."""


ObservabilityProviderConfig = Annotated[
    LogfireProviderConfig | AgentOpsProviderConfig, Field(discriminator="type")
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
