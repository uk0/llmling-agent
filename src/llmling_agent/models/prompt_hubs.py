"""Prompt models for agent configuration."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr


class PromptHubConfig(BaseModel):
    """Configuration for prompt providers."""

    type: str = Field(init=False)
    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)


class PromptLayerConfig(PromptHubConfig):
    type: Literal["promptlayer"] = Field("promptlayer", init=False)
    api_key: SecretStr


class OpenLITConfig(PromptHubConfig):
    """Configuration for OpenLIT prompt provider."""

    type: Literal["openlit"] = Field("openlit", init=False)
    url: str | None = None  # Optional, defaults to OPENLIT_URL env var
    api_key: SecretStr | None = None  # Optional, defaults to OPENLIT_API_KEY env var


class LangfuseConfig(PromptHubConfig):
    """Configuration for Langfuse prompt provider."""

    type: Literal["langfuse"] = Field("langfuse", init=False)
    secret_key: SecretStr
    public_key: SecretStr
    host: str = "https://cloud.langfuse.com"
    cache_ttl_seconds: int = 60
    max_retries: int = 2
    fetch_timeout_seconds: int = 20
