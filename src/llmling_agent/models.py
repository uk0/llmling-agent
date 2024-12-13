"""Models for agent configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Self

from llmling import Config
from llmling.config.models import GlobalSettings, LLMCapabilitiesConfig
from llmling.config.store import ConfigStore
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_ai import models  # noqa: TC002
from upath.core import UPath
import yamling

from llmling_agent.config.capabilities import BUILTIN_ROLES, Capabilities, RoleName
from llmling_agent.log import get_logger


logger = get_logger(__name__)

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

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")


class ResponseDefinition(BaseModel):
    """Definition of an agent response type."""

    description: str | None = None
    """Optional description of the response type"""
    fields: dict[str, ResponseField]
    """Mapping of field names to their definitions"""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")


class SystemPrompt(BaseModel):
    """System prompt configuration."""

    type: Literal["text", "function", "template"]
    """Type of system prompt: static text, function call, or template"""
    value: str
    """The prompt text, function path, or template string"""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")


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

    result_type: str | None = None  # References a response name or None for plain string
    """Name of the response definition to use"""

    avatar: str | None = None
    """URL or path to agent's avatar image"""

    system_prompts: list[str] = Field(default_factory=list)
    """System prompts for the agent"""

    user_prompts: list[str] = Field(default_factory=list)
    """Default user prompts for the agent"""

    model_settings: dict[str, Any] = Field(default_factory=dict)
    """Additional settings to pass to the model"""

    config_file_path: str | None = None
    """Config file path for resolving environment."""

    role: RoleName = "assistant"
    """Role name (built-in or custom) determining agent's capabilities."""

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        extra="forbid",
        use_attribute_docstrings=True,
    )

    @staticmethod
    def _resolve_environment_path(env: str, config_file_path: str | None = None) -> str:
        """Resolve environment path from config store or relative path."""
        try:
            config_store = ConfigStore()
            return config_store.get_config(env)
        except KeyError:
            if config_file_path:
                base_dir = UPath(config_file_path).parent
                return str(base_dir / env)
            return env

    @model_validator(mode="before")
    @classmethod
    def resolve_paths(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Resolve environment path before model creation."""
        if env := data.get("environment"):
            config_path = data.get("config_file_path")
            data["environment"] = cls._resolve_environment_path(env, config_path)
            # Store the config path for later use in get_config
            data["config_file_path"] = config_path
        return data

    def get_config(self) -> Config:
        """Get configuration for this agent."""
        if self.environment:
            cfg_path = getattr(self, "config_file_path", None)
            path = self._resolve_environment_path(self.environment, cfg_path)
            return Config.from_file(path)

        caps = LLMCapabilitiesConfig(load_resource=False, get_resources=False)
        global_settings = GlobalSettings(llm_capabilities=caps)
        return Config(global_settings=global_settings)

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


class AgentsManifest(BaseModel):
    """Complete agent definition including responses."""

    responses: dict[str, ResponseDefinition]
    """Mapping of response names to their definitions"""
    agents: dict[str, AgentConfig]
    """Mapping of agent IDs to their configurations"""
    roles: dict[str, Capabilities] = Field(default_factory=dict)
    """Custom role definitions"""

    model_config = ConfigDict(
        use_attribute_docstrings=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    @model_validator(mode="after")
    def validate_response_types(self) -> AgentsManifest:
        """Ensure all agent result_types exist in responses."""
        for agent_id, agent in self.agents.items():
            if agent.result_type not in self.responses:
                msg = f"'{agent.result_type=}' for '{agent_id=}' not found in responses"
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
            # Set identifier as name if not set
            for identifier, config in data["agents"].items():
                if not config.get("name"):
                    config["name"] = identifier
            agent_def = cls.model_validate(data)
            # Update all agents with the config file path and ensure names
            agents = {
                name: config.model_copy(update={"config_file_path": str(path)})
                for name, config in agent_def.agents.items()
            }
            return agent_def.model_copy(update={"agents": agents})
        except Exception as exc:
            msg = f"Failed to load agent config from {path}"
            raise ValueError(msg) from exc

    def get_capabilities(self, role_name: RoleName) -> Capabilities:
        """Get capabilities for a role name.

        Args:
            role_name: Either a built-in role or custom role name

        Returns:
            Capability configuration for the role

        Raises:
            ValueError: If role doesn't exist
        """
        if role_name in self.roles:
            return self.roles[role_name]
        if isinstance(role_name, str) and role_name in BUILTIN_ROLES:
            return BUILTIN_ROLES[role_name]  # type: ignore[index]
        msg = f"Unknown role: {role_name}"
        raise ValueError(msg)


@dataclass
class ResourceInfo:
    """Information about an available resource.

    This class provides essential information about a resource that can be loaded.
    Use the resource name with load_resource() to access the actual content.
    """

    name: str
    """Name of the resource, use this with load_resource()"""

    uri: str
    """URI identifying the resource location"""

    description: str | None = None
    """Optional description of the resource's content or purpose"""
