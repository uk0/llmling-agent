"""Models for agent configuration."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
from typing import TYPE_CHECKING, Any, Self, overload

from llmling import (
    Config,
    ConfigModel,
    ConfigStore,
    GlobalSettings,
    LLMCapabilitiesConfig,
)
from llmling_models.model_types import AnyModel  # noqa: TC002
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_ai.agent import EndStrategy  # noqa: TC002
from pydantic_ai.models.test import TestModel
from typing_extensions import TypeVar
from upath.core import UPath
import yamling

from llmling_agent.config import Capabilities
from llmling_agent.environment import AgentEnvironment, FileEnvironment, InlineEnvironment
from llmling_agent.events.sources import EventConfig  # noqa: TC001
from llmling_agent.models.forward_targets import ForwardingTarget  # noqa: TC001
from llmling_agent.models.sources import ContextSource, Knowledge  # noqa: TC001
from llmling_agent.responses import InlineResponseDefinition, ResponseDefinition
from llmling_agent.templating import render_prompt


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from uuid import UUID

    from llmling_agent import AgentPool, LLMlingAgent
    from llmling_agent.common_types import StrPath


TDeps = TypeVar("TDeps", default=Any)
TResult = TypeVar("TResult", default=Any)
TResultOverride = TypeVar("TResultOverride")


class WorkerConfig(BaseModel):
    """Configuration for a worker agent."""

    name: str
    reset_history_on_run: bool = True
    pass_message_history: bool = False
    share_context: bool = False

    @classmethod
    def from_str(cls, name: str) -> WorkerConfig:
        """Create config from simple string form."""
        return cls(name=name)


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

    inherits: str | None = None
    """Name of agent config to inherit from"""

    description: str | None = None
    """Optional description of the agent's purpose"""

    model: str | AnyModel | None = None
    """The model to use for this agent. Can be either a simple model name
    string (e.g. 'openai:gpt-4') or a structured model definition."""

    environment: str | AgentEnvironment | None = None
    """Environment configuration (path or object)"""

    capabilities: Capabilities = Field(default_factory=Capabilities)
    """Current agent's capabilities."""

    session_id: str | None = None
    """Opetional id of a session to load."""

    result_type: str | ResponseDefinition | None = None
    """Name of the response definition to use"""

    retries: int = 1
    """Number of retries for failed operations (maps to pydantic-ai's retries)"""

    result_tool_name: str = "final_result"
    """Name of the tool used for structured responses"""

    result_tool_description: str | None = None
    """Custom description for the result tool"""

    result_retries: int | None = None
    """Max retries for result validation"""

    end_strategy: EndStrategy = "early"
    """The strategy for handling multiple tool calls when a final result is found"""

    # defer_model_check: bool = False
    # """Whether to defer model evaluation until first run"""

    avatar: str | None = None
    """URL or path to agent's avatar image"""

    system_prompts: list[str] = Field(default_factory=list)
    """System prompts for the agent"""

    user_prompts: list[str] = Field(default_factory=list)
    """Default user prompts for the agent"""

    context_sources: list[ContextSource] = Field(default_factory=list)
    """Initial context sources to load"""

    include_role_prompts: bool = True
    """Whether to include default prompts based on the agent's role."""

    model_settings: dict[str, Any] = Field(default_factory=dict)
    """Additional settings to pass to the model"""

    config_file_path: str | None = None
    """Config file path for resolving environment."""

    triggers: list[EventConfig] = Field(default_factory=list)
    """Event sources that activate this agent"""

    knowledge: Knowledge | None = None
    """Knowledge sources for this agent."""

    forward_to: list[ForwardingTarget] = Field(default_factory=list)
    """Targets to forward results to."""

    workers: list[WorkerConfig] = Field(default_factory=list)
    """Worker agents which will be available as tools."""

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        extra="forbid",
        use_attribute_docstrings=True,
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_workers(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Convert string workers to WorkerConfig."""
        if workers := data.get("workers"):
            data["workers"] = [
                WorkerConfig.from_str(w)
                if isinstance(w, str)
                else w
                if isinstance(w, WorkerConfig)  # Keep existing WorkerConfig
                else WorkerConfig(**w)  # Convert dict to WorkerConfig
                for w in workers
            ]
        return data

    @model_validator(mode="before")
    @classmethod
    def validate_result_type(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Convert result type to proper format."""
        result_type = data.get("result_type")
        if isinstance(result_type, dict):
            # Convert inline definition to ResponseDefinition
            if "type" not in result_type:
                # Default to inline type if not specified
                result_type["type"] = "inline"
            data["result_type"] = InlineResponseDefinition(**result_type)
        return data

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

    def render_system_prompts(self, context: dict[str, Any] | None = None) -> list[str]:
        """Render system prompts with context."""
        if not context:
            # Default context
            context = {"name": self.name, "id": 1, "model": self.model}
        return [render_prompt(p, {"agent": context}) for p in self.system_prompts]

    def get_config(self) -> Config:
        """Get configuration for this agent."""
        match self.environment:
            case None:
                # Create minimal config
                caps = LLMCapabilitiesConfig()
                global_settings = GlobalSettings(llm_capabilities=caps)
                return Config(global_settings=global_settings)
            case str() as path:
                # Backward compatibility: treat as file path
                resolved = self._resolve_environment_path(path, self.config_file_path)
                return Config.from_file(resolved)
            case FileEnvironment(uri=uri) as env:
                # Handle FileEnvironment instance
                resolved = env.get_file_path()
                return Config.from_file(resolved)
            case {"type": "file", "uri": uri}:
                # Handle raw dict matching file environment structure
                return Config.from_file(uri)
            case {"type": "inline", "config": config}:
                return config
            case InlineEnvironment() as config:
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
            "description": self.description,
            "model": self.model,
            "system_prompt": self.system_prompts,
            "retries": self.retries,
            "result_tool_name": self.result_tool_name,
            "session_id": self.session_id,
            "result_tool_description": self.result_tool_description,
            "result_retries": self.result_retries,
            "end_strategy": self.end_strategy,
            # "defer_model_check": self.defer_model_check,
            **self.model_settings,
        }
        # Note: result_type is handled separately as it needs to be resolved
        # from string to actual type in LLMlingAgent initialization

        dct.update(overrides)
        return dct


# TODO: python 3.13: set defaults here
class AgentsManifest[TDeps, TResult](ConfigModel):
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

    model_config = ConfigDict(
        use_attribute_docstrings=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    def clone_agent_config(
        self,
        name: str,
        new_name: str | None = None,
        *,
        template_context: dict[str, Any] | None = None,
        **overrides: Any,
    ) -> str:
        """Create a copy of an agent configuration.

        Args:
            name: Name of agent to clone
            new_name: Optional new name (auto-generated if None)
            template_context: Variables for template rendering
            **overrides: Configuration overrides for the clone

        Returns:
            Name of the new agent

        Raises:
            KeyError: If original agent not found
            ValueError: If new name already exists or if overrides invalid
        """
        if name not in self.agents:
            msg = f"Agent {name} not found"
            raise KeyError(msg)

        actual_name = new_name or f"{name}_copy_{len(self.agents)}"
        if actual_name in self.agents:
            msg = f"Agent {actual_name} already exists"
            raise ValueError(msg)

        # Deep copy the configuration
        config = self.agents[name].model_copy(deep=True)

        # Apply overrides
        for key, value in overrides.items():
            if not hasattr(config, key):
                msg = f"Invalid override: {key}"
                raise ValueError(msg)
            setattr(config, key, value)

        # Handle template rendering if context provided
        if template_context:
            # Apply name from context if not explicitly overridden
            if "name" in template_context and "name" not in overrides:
                config.name = template_context["name"]

            # Render system prompts
            config.system_prompts = config.render_system_prompts(template_context)

        self.agents[actual_name] = config
        return actual_name

    @model_validator(mode="before")
    @classmethod
    def resolve_inheritance(cls, data: dict) -> dict:
        """Resolve agent inheritance chains."""
        agents = data.get("agents", {})
        resolved: dict[str, dict] = {}
        seen: set[str] = set()

        def resolve_agent(name: str) -> dict:
            if name in resolved:
                return resolved[name]

            if name in seen:
                msg = f"Circular inheritance detected: {name}"
                raise ValueError(msg)

            seen.add(name)
            config = (
                agents[name].model_copy()
                if hasattr(agents[name], "model_copy")
                else agents[name].copy()
            )
            inherit = (
                config.get("inherits") if isinstance(config, dict) else config.inherits
            )
            if inherit:
                if inherit not in agents:
                    msg = f"Parent agent {inherit} not found"
                    raise ValueError(msg)

                # Get resolved parent config
                parent = resolve_agent(inherit)
                # Merge parent with child (child overrides parent)
                merged = parent.copy()
                merged.update(config)
                config = merged

            seen.remove(name)
            resolved[name] = config
            return config

        # Resolve all agents
        for name in agents:
            resolved[name] = resolve_agent(name)

        # Update agents with resolved configs
        data["agents"] = resolved
        return data

    @model_validator(mode="after")
    def validate_response_types(self) -> AgentsManifest:
        """Ensure all agent result_types exist in responses or are inline."""
        for agent_id, agent in self.agents.items():
            if (
                isinstance(agent.result_type, str)
                and agent.result_type not in self.responses
            ):
                msg = f"'{agent.result_type=}' for '{agent_id=}' not found in responses"
                raise ValueError(msg)
        return self

    @classmethod
    def from_file(cls, path: StrPath) -> Self:
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

    async def create_pool(
        self,
        *,
        agents_to_load: list[str] | None = None,
        connect_agents: bool = True,
        session_id: str | UUID | None = None,
    ) -> AgentPool:
        """Create an agent pool from this manifest.

        Args:
            agents_to_load: Optional list of agents to initialize
            connect_agents: Whether to set up forwarding connections
            session_id: Optional session ID for conversation recovery

        Returns:
            Configured agent pool
        """
        from llmling_agent.delegation import AgentPool

        pool = AgentPool(
            manifest=self,
            agents_to_load=agents_to_load,
            connect_agents=connect_agents,
        )

        # Initialize agents with knowledge
        for name, agent in pool.agents.items():
            if (config := self.agents.get(name)) and config.knowledge:
                for source in (
                    config.knowledge.paths
                    + config.knowledge.resources
                    + config.knowledge.prompts
                ):
                    await agent.conversation.load_context_source(source)  # type: ignore

        return pool

    @overload
    async def open_agent(
        self,
        agent_name: str,
        *,
        return_type: None = None,
        model: str | None = None,
        session_id: str | UUID | None = None,
    ) -> AsyncIterator[LLMlingAgent[TDeps, TResult]]: ...

    @overload
    async def open_agent(
        self,
        agent_name: str,
        *,
        return_type: type[TResultOverride],
        model: str | None = None,
        session_id: str | UUID | None = None,
    ) -> AsyncIterator[LLMlingAgent[TDeps, TResultOverride]]: ...

    @asynccontextmanager
    async def open_agent(
        self,
        agent_name: str,
        *,
        return_type: type[TResultOverride] | None = None,
        model: str | None = None,
        session_id: str | UUID | None = None,
    ) -> AsyncIterator[LLMlingAgent[TDeps, TResult | TResultOverride]]:
        """Open and configure a specific agent from configuration.

        Creates the agent in the context of a single-agent pool.

        Args:
            agent_name: Name of the agent to load
            return_type: Optional type override for agent's return type
            model: Optional model override
            session_id: Optional ID to recover a previous state

        Example:
            manifest = AgentsManifest[Any, str].from_file("agents.yml")
            async with manifest.open_agent("my-agent") as agent:
                result = await agent.run("Hello!")
        """
        from llmling_agent import LLMlingAgent
        from llmling_agent.delegation import AgentPool

        # Create empty pool just for context
        pool = AgentPool(manifest=self, agents_to_load=[], connect_agents=False)
        try:
            async with LLMlingAgent[TDeps, TResult | TResultOverride].open_agent(
                self,
                agent_name,
                model=model,
                session_id=session_id,
                result_type=return_type,
            ) as agent:
                if agent._context:
                    agent._context.pool = pool
                pool.agents[agent_name] = agent
                yield agent
        finally:
            await pool.cleanup()


class ToolCallInfo(BaseModel):
    """Information about an executed tool call."""

    tool_name: str
    """Name of the tool that was called."""

    args: dict[str, Any]
    """Arguments passed to the tool."""

    result: Any
    """Result returned by the tool."""

    tool_call_id: str | None
    """ID provided by the model (e.g. OpenAI function call ID)."""

    timestamp: datetime = Field(default_factory=datetime.now)
    """When the tool was called."""

    message_id: str | None = None
    """ID of the message that triggered this tool call."""

    context_data: Any | None = None
    """Optional context data that was passed to the agent's run() method."""

    error: str | None = None
    """Error message if the tool call failed."""

    timing: float | None = None
    """Time taken for this specific tool call in seconds."""

    agent_tool_name: str | None = None
    """If this tool is agent-based, the name of that agent."""

    model_config = ConfigDict(use_attribute_docstrings=True)
