"""Models for agent configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Self

from llmling import Config
from llmling.config.models import GlobalSettings, LLMCapabilitiesConfig
from llmling.config.store import ConfigStore
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic_ai import models  # noqa: TC002
from upath import UPath
import yamling


if TYPE_CHECKING:
    import os


class ResponseField(BaseModel):
    """Field definition for agent responses."""

    type: str
    """Data type of the response field"""
    description: str | None = None
    """Optional description of what this field represents"""
    constraints: dict[str, Any] | None = None
    """Optional validation constraints for the field"""

    model_config = ConfigDict(
        use_attribute_docstrings=True,
        extra="forbid",
    )


class ResponseDefinition(BaseModel):
    """Definition of an agent response type."""

    description: str | None = None
    """Optional description of the response type"""
    fields: dict[str, ResponseField]
    """Mapping of field names to their definitions"""

    model_config = ConfigDict(
        use_attribute_docstrings=True,
        extra="forbid",
    )


class SystemPrompt(BaseModel):
    """System prompt configuration."""

    type: Literal["text", "function", "template"]
    """Type of system prompt: static text, function call, or template"""
    value: str
    """The prompt text, function path, or template string"""

    model_config = ConfigDict(
        use_attribute_docstrings=True,
        extra="forbid",
    )


class AgentConfig(BaseModel):
    """Configuration for a single agent."""

    name: str | None = None
    """Name of the agent"""

    description: str | None = None
    """Optional description of the agent's purpose"""

    model: models.Model | models.KnownModelName | None = None
    """The LLM model to use"""

    environment: str | None = None
    """Path or name of the environment configuration to use"""

    result_type: str
    """Name of the response definition to use"""

    system_prompts: list[str] = Field(default_factory=list)
    """System prompts for the agent"""

    user_prompts: list[str] = Field(default_factory=list)
    """Default user prompts for the agent"""

    model_settings: dict[str, Any] = Field(default_factory=dict)
    """Additional settings to pass to the model"""

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        extra="forbid",
        use_attribute_docstrings=True,
    )

    def get_config(self) -> Config:
        """Get configuration for this agent.

        Returns:
            Config: If environment is set, loaded from that file.
                   Otherwise, returns minimal config with resource tools disabled.
        """
        if self.environment:
            return Config.from_file(self.environment)

        caps = LLMCapabilitiesConfig(load_resource=False, get_resources=False)
        global_settings = GlobalSettings(llm_capabilities=caps)
        return Config(global_settings=global_settings)

    @field_validator("environment")
    @classmethod
    def resolve_environment(cls, env: str, info: ValidationInfo) -> str | None:
        """Try to resolve environment as name first, then as path."""
        if not env:
            return None

        try:
            # First try as stored config name
            config_store = ConfigStore()
            return config_store.get_config(env)
        except KeyError:
            # If not found, treat as relative path to agent config
            if "config_file_path" in info.data:
                base_dir = UPath(info.data["config_file_path"]).parent
                return str(base_dir / env)
            return env

    def get_agent_kwargs(self, **overrides) -> dict[str, Any]:
        """Get kwargs for LLMlingAgent constructor.

        Returns:
            dict[str, Any]: Kwargs to pass to LLMlingAgent
        """
        # Include only the fields that LLMlingAgent expects
        dct = {
            "name": self.name,
            "model": self.model,
            # "result_type": self.result_type,
            "system_prompt": self.system_prompts,
            **self.model_settings,
        }
        dct.update(overrides)
        return dct


class AgentDefinition(BaseModel):
    """Complete agent definition including responses."""

    responses: dict[str, ResponseDefinition]
    """Mapping of response names to their definitions"""
    agents: dict[str, AgentConfig]
    """Mapping of agent IDs to their configurations"""

    model_config = ConfigDict(
        use_attribute_docstrings=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    @model_validator(mode="after")
    def validate_response_types(self) -> AgentDefinition:
        """Ensure all agent result_types exist in responses."""
        for agent_id, agent in self.agents.items():
            if agent.result_type not in self.responses:
                msg = (
                    f"Response type '{agent.result_type}' for agent '{agent_id}' "
                    "not found in responses"
                )
                raise ValueError(msg)
        return self

    @classmethod
    def from_file(cls, path: str | os.PathLike[str]) -> Self:
        """Load agent configuration from YAML file.

        Args:
            path: Path to the configuration file

        Returns:
            Loaded agent definition

        Raises:
            ValueError: If loading fails
        """
        try:
            data = yamling.load_yaml_file(path)
            return cls.model_validate(
                data,
                context={"config_file_path": str(path)},
            )
        except Exception as exc:
            msg = f"Failed to load agent config from {path}"
            raise ValueError(msg) from exc
