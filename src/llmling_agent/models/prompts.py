"""Prompt models for agent configuration."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


SystemPromptCategory = Literal["role", "methodology", "quality", "task"]


class SystemPrompt(BaseModel):
    """Individual system prompt definition."""

    content: str
    """The actual prompt text."""

    type: SystemPromptCategory = "role"
    """Categorization for template organization."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)


class PromptConfig(BaseModel):
    """Complete prompt configuration."""

    system_prompts: dict[str, SystemPrompt] = Field(default_factory=dict)
    """Mapping of system prompt identifiers to their definitions."""

    template: str | None = None
    """Optional template for combining prompts.
    Has access to prompts grouped by type."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)
