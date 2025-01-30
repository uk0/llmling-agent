"""Prompt models for agent configuration."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr


class PromptHubConfig(BaseModel):
    """Configuration for prompt providers."""

    type: str = Field(init=False)
    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True, extra="forbid")


class PromptLayerConfig(PromptHubConfig):
    type: Literal["promptlayer"] = Field("promptlayer", init=False)
    """Configuration for PromptLayer prompt provider."""

    api_key: SecretStr
    """API key for the PromptLayer API."""


class OpenLITConfig(PromptHubConfig):
    """Configuration for OpenLIT prompt provider."""

    type: Literal["openlit"] = Field("openlit", init=False)
    """Configuration for OpenLIT prompt provider."""

    url: str | None = None  # Optional, defaults to OPENLIT_URL env var
    """URL of the OpenLIT API."""

    api_key: SecretStr | None = None  # Optional, defaults to OPENLIT_API_KEY env var
    """API key for the OpenLIT API."""


class LangfuseConfig(PromptHubConfig):
    """Configuration for Langfuse prompt provider."""

    type: Literal["langfuse"] = Field("langfuse", init=False)
    """Configuration for Langfuse prompt provider."""

    secret_key: SecretStr
    """Secret key for the Langfuse API."""

    public_key: SecretStr
    """Public key for the Langfuse API."""

    host: str = "https://cloud.langfuse.com"
    """Langfuse host address."""

    cache_ttl_seconds: int = 60
    """Cache TTL for responses in seconds."""

    max_retries: int = 2
    """Maximum number of retries for failed requests."""

    fetch_timeout_seconds: int = 20
    """Timeout for fetching responses in seconds."""
