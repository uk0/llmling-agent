"""Prompt models for agent configuration."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PromptHubConfig(BaseModel):
    """Configuration for prompt providers."""

    type: str = Field(init=False)
    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)


class PromptLayerConfig(PromptHubConfig):
    type: Literal["promptlayer"] = Field("promptlayer", init=False)
    api_key: str


class OpenLITConfig(PromptHubConfig):
    """Configuration for OpenLIT prompt provider."""

    type: Literal["openlit"] = Field("openlit", init=False)
    url: str | None = None  # Optional, defaults to OPENLIT_URL env var
    api_key: str | None = None  # Optional, defaults to OPENLIT_API_KEY env var


class HuggingFaceConfig(PromptHubConfig):
    """Configuration for HuggingFace prompt provider."""

    type: Literal["huggingface"] = Field("huggingface", init=False)
    api_key: str | None = None
    base_url: str | None = None
    workspace: str | None = None
