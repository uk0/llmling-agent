"""Models for agent configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import models  # noqa: TC002
import yamling


if TYPE_CHECKING:
    import os


class ResponseField(BaseModel):
    """Field definition for agent responses."""

    type: str
    description: str | None = None
    constraints: dict[str, Any] | None = None


class ResponseDefinition(BaseModel):
    """Definition of an agent response type."""

    description: str | None = None
    fields: dict[str, ResponseField]


class SystemPrompt(BaseModel):
    """System prompt configuration."""

    type: Literal["text", "function", "template"]
    value: str


class AgentConfig(BaseModel):
    """Configuration for a single agent."""

    name: str
    description: str | None = None
    model: models.Model | models.KnownModelName | None = None
    model_settings: dict[str, Any] = Field(default_factory=dict)
    result_model: str
    deps_type: str | None = None
    system_prompts: list[SystemPrompt]

    model_config = ConfigDict(arbitrary_types_allowed=True)


class AgentDefinition(BaseModel):
    """Complete agent definition including responses."""

    responses: dict[str, ResponseDefinition]
    agents: dict[str, AgentConfig]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_file(cls, path: str | os.PathLike[str]) -> Self:
        """Load agent configuration from YAML file."""
        try:
            data = yamling.load_yaml_file(path)
            return cls.model_validate(data)
        except Exception as exc:
            msg = f"Failed to load agent config from {path}"
            raise ValueError(msg) from exc
