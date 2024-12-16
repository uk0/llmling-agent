"""Models for agent configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, Literal, Self

from llmling import Config
from llmling.config.models import ConfigModel, GlobalSettings, LLMCapabilitiesConfig
from llmling.config.store import ConfigStore
from llmling.utils import importing
from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator
from pydantic_ai.models.test import TestModel
from upath.core import UPath
import yamling

from llmling_agent.config.capabilities import BUILTIN_ROLES, Capabilities, RoleName
from llmling_agent.environment import AgentEnvironment  # noqa: TC001
from llmling_agent.environment.models import FileEnvironment, InlineEnvironment
from llmling_agent.log import get_logger
from llmling_agent.pydanticai_models.types import AnyModel  # noqa: TC001


if TYPE_CHECKING:
    import os

    from llmling_agent.context import AgentContext


TYPE_MAP = {
    "str": str,
    "bool": bool,
    "int": int,
    "float": float,
    "list[str]": list[str],
}

logger = get_logger(__name__)


def resolve_response_type(
    type_name: str,
    context: AgentContext | None,
) -> type[BaseModel]:
    """Resolve response type from string name to actual type.

    Args:
        type_name: Name of the response type
        context: Agent context containing response definitions

    Returns:
        Resolved Pydantic model type

    Raises:
        ValueError: If type cannot be resolved
    """
    if not context or type_name not in context.definition.responses:
        msg = f"Result type {type_name} not found in responses"
        raise ValueError(msg)

    response_def = context.definition.responses[type_name]
    match response_def:
        case ImportedResponseDefinition():
            return response_def.resolve_model()
        case InlineResponseDefinition():
            # Create Pydantic model from inline definition
            fields = {}
            for name, field in response_def.fields.items():
                python_type = TYPE_MAP.get(field.type)
                if not python_type:
                    msg = f"Unsupported field type: {field.type}"
                    raise ValueError(msg)

                field_info = Field(description=field.description)
                fields[name] = (python_type, field_info)
            cls_name = response_def.description or "ResponseType"
            return create_model(cls_name, **fields, __base__=BaseModel)  # type: ignore[call-overload]
        case _:
            msg = f"Unknown response definition type: {type(response_def)}"
            raise ValueError(msg)


class ResponseField(BaseModel):
    """Field definition for inline response types.

    Defines a single field in an inline response definition, including:
    - Data type specification
    - Optional description
    - Validation constraints

    Used by InlineResponseDefinition to structure response fields.
    """

    type: str
    """Data type of the response field"""
    description: str | None = None
    """Optional description of what this field represents"""
    constraints: dict[str, Any] | None = None
    """Optional validation constraints for the field"""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")


class InlineResponseDefinition(BaseModel):
    """Inline definition of an agent's response structure.

    Allows defining response types directly in the configuration using:
    - Field definitions with types and descriptions
    - Optional validation constraints
    - Custom field descriptions

    Example:
        responses:
          BasicResult:
            type: inline
            fields:
              success: {type: bool, description: "Operation success"}
              message: {type: str, description: "Result details"}
    """

    type: Literal["inline"] = Field("inline", init=False)
    description: str | None = None
    fields: dict[str, ResponseField]
    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")


class ImportedResponseDefinition(BaseModel):
    """Response definition that imports an existing Pydantic model.

    Allows using externally defined Pydantic models as response types.
    Benefits:
    - Reuse existing model definitions
    - Full Python type support
    - Complex validation logic
    - IDE support for imported types

    Example:
        responses:
          AnalysisResult:
            type: import
            import_path: myapp.models.AnalysisResult
    """

    type: Literal["import"] = Field("import", init=False)
    description: str | None = None
    import_path: str
    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    # mypy is confused about
    def resolve_model(self) -> type[BaseModel]:  # type: ignore
        """Import and return the model class."""
        try:
            model_class = importing.import_class(self.import_path)
            if not issubclass(model_class, BaseModel):
                msg = f"{self.import_path} must be a Pydantic model"
                raise TypeError(msg)  # noqa: TRY301
        except Exception as e:
            msg = f"Failed to import response type {self.import_path}"
            raise ValueError(msg) from e
        else:
            return model_class


ResponseDefinition = Annotated[
    InlineResponseDefinition | ImportedResponseDefinition, Field(discriminator="type")
]


class SystemPrompt(BaseModel):
    """System prompt configuration for agent behavior control.

    Defines prompts that set up the agent's behavior and context.
    Supports multiple types:
    - Static text prompts
    - Dynamic function-based prompts
    - Template prompts with variable substitution
    """

    type: Literal["text", "function", "template"]
    """Type of system prompt: static text, function call, or template"""
    value: str
    """The prompt text, function path, or template string"""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")


class AgentConfig(BaseModel):
    """Configuration for a single agent in the system.

    Defines an agent's complete configuration including its model, environment,
    capabilities, and behavior settings. Each agent can have its own:
    - Language model configuration
    - Environment setup (tools and resources)
    - Response type definitions
    - System prompts and default user prompts
    - Role-based capabilities

    The configuration can be loaded from YAML or created programmatically.
    """

    name: str | None = None
    """Name of the agent"""

    description: str | None = None
    """Optional description of the agent's purpose"""

    model: str | AnyModel | None = None
    """The model to use for this agent. Can be either a simple model name
    string (e.g. 'openai:gpt-4') or a structured model definition."""

    environment: str | AgentEnvironment | None = None
    """Environment configuration (path or object)"""

    result_type: str | None = None
    """Name of the response definition to use"""

    retries: int = 1
    """Number of retries for failed operations (maps to pydantic-ai's retries)"""

    result_tool_name: str = "final_result"
    """Name of the tool used for structured responses"""

    result_tool_description: str | None = None
    """Custom description for the result tool"""

    result_retries: int | None = None
    """Max retries for result validation"""

    # defer_model_check: bool = False
    # """Whether to defer model evaluation until first run"""

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

    @model_validator(mode="before")
    @classmethod
    def handle_model_types(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Convert model inputs to appropriate format."""
        model = data.get("model")
        match model:
            case str():
                data["model"] = {"type": "string", "identifier": model}
            case TestModel():
                # Wrap TestModel in our custom wrapper
                data["model"] = {"type": "test", "model": model}
        return data

    def get_config(self) -> Config:
        """Get configuration for this agent."""
        match self.environment:
            case None:
                # Create minimal config
                caps = LLMCapabilitiesConfig(load_resource=False, get_resources=False)
                global_settings = GlobalSettings(llm_capabilities=caps)
                return Config(global_settings=global_settings)
            case str() as path:
                # Backward compatibility: treat as file path
                resolved = self._resolve_environment_path(path, self.config_file_path)
                return Config.from_file(resolved)
            case {"type": "file", "uri": uri} | FileEnvironment(uri=uri):
                # Handle file environment
                return Config.from_file(uri)
            case {"type": "inline", "config": config} | InlineEnvironment(config=config):
                # Handle inline environment
                return config
            case _:
                msg = f"Invalid environment configuration: {self.environment}"
                raise ValueError(msg)

    def get_environment_path(self) -> str | None:
        """Get environment file path if available."""
        match self.environment:
            case str() as path:
                return self._resolve_environment_path(path, self.config_file_path)
            case {"type": "file", "uri": uri} | FileEnvironment(uri=uri):
                return uri
            case _:
                return None

    def get_environment_display(self) -> str:
        """Get human-readable environment description."""
        match self.environment:
            case str() as path:
                return f"File: {path}"
            case {"type": "file", "uri": uri} | FileEnvironment(uri=uri):
                return f"File: {uri}"
            case {"type": "inline", "uri": uri} | InlineEnvironment(uri=uri) if uri:
                return f"Inline: {uri}"
            case {"type": "inline"} | InlineEnvironment():
                return "Inline configuration"
            case None:
                return "No environment configured"
            case _:
                return "Invalid environment configuration"

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
        """Store config file path for later use."""
        if "environment" in data:
            # Just store the config path for later use
            data["config_file_path"] = data.get("config_file_path")
        return data

    def get_agent_kwargs(self, **overrides) -> dict[str, Any]:
        """Get kwargs for LLMlingAgent constructor.

        Returns:
            dict[str, Any]: Kwargs to pass to LLMlingAgent
        """
        # Include only the fields that LLMlingAgent expects
        dct = {
            "name": self.name,
            "model": self.model,
            "system_prompt": self.system_prompts,
            "retries": self.retries,
            "result_tool_name": self.result_tool_name,
            "result_tool_description": self.result_tool_description,
            "result_retries": self.result_retries,
            # "defer_model_check": self.defer_model_check,
            **self.model_settings,
        }
        # Note: result_type is handled separately as it needs to be resolved
        # from string to actual type in LLMlingAgent initialization

        dct.update(overrides)
        return dct


class AgentsManifest(ConfigModel):
    """Complete agent configuration manifest defining all available agents.

    This is the root configuration that:
    - Defines available response types (both inline and imported)
    - Configures all agent instances and their settings
    - Sets up custom role definitions and capabilities
    - Manages environment configurations

    A single manifest can define multiple agents that can work independently
    or collaborate through the orchestrator.
    """

    responses: dict[str, ResponseDefinition] = Field(default_factory=dict)
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
            if agent.result_type is not None and agent.result_type not in self.responses:
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
