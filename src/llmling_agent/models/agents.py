"""Models for agent configuration."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID

from llmling import (
    BasePrompt,
    Config,
    ConfigStore,
    GlobalSettings,
    LLMCapabilitiesConfig,
    PromptMessage,
    StaticPrompt,
)
from llmling.utils.importing import import_callable, import_class
from pydantic import Field, model_validator
from toprompt import render_prompt

from llmling_agent.common_types import EndStrategy  # noqa: TC001
from llmling_agent.config import Capabilities
from llmling_agent.resource_providers.static import StaticResourceProvider
from llmling_agent_config.environment import (
    AgentEnvironment,
    FileEnvironment,
    InlineEnvironment,
)
from llmling_agent_config.knowledge import Knowledge  # noqa: TC001
from llmling_agent_config.nodes import NodeConfig
from llmling_agent_config.providers import ProviderConfig  # noqa: TC001
from llmling_agent_config.result_types import InlineResponseDefinition, ResponseDefinition
from llmling_agent_config.session import MemoryConfig, SessionQuery
from llmling_agent_config.tools import BaseToolConfig, ToolConfig
from llmling_agent_config.toolsets import ToolsetConfig  # noqa: TC001
from llmling_agent_config.workers import WorkerConfig  # noqa: TC001
from llmling_agent_models import AnyModelConfig  # noqa: TC001


if TYPE_CHECKING:
    from llmling_agent.resource_providers.base import ResourceProvider
    from llmling_agent.tools.base import Tool
    from llmling_agent_providers.base import AgentProvider


ToolConfirmationMode = Literal["always", "never", "per_tool"]
ProviderName = Literal["pydantic_ai", "human", "litellm"]

logger = logging.getLogger(__name__)


class AgentConfig(NodeConfig):
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

    provider: ProviderConfig | ProviderName = "pydantic_ai"
    """Provider configuration or shorthand type"""

    inherits: str | None = None
    """Name of agent config to inherit from"""

    model: str | AnyModelConfig | None = None
    """The model to use for this agent. Can be either a simple model name
    string (e.g. 'openai:gpt-4') or a structured model definition."""

    tools: list[ToolConfig | str] = Field(default_factory=list)
    """A list of tools to register with this agent."""

    toolsets: list[ToolsetConfig] = Field(default_factory=list)
    """Toolset configurations for extensible tool collections."""

    environment: str | AgentEnvironment | None = None
    """Environments configuration (path or object)"""

    capabilities: Capabilities = Field(default_factory=Capabilities)
    """Current agent's capabilities."""

    session: str | SessionQuery | MemoryConfig | None = None
    """Session configuration for conversation recovery."""

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

    avatar: str | None = None
    """URL or path to agent's avatar image"""

    system_prompts: list[str] = Field(default_factory=list)
    """System prompts for the agent"""

    library_system_prompts: list[str] = Field(default_factory=list)
    """System prompts for the agent from the library"""

    user_prompts: list[str] = Field(default_factory=list)
    """Default user prompts for the agent"""

    # context_sources: list[ContextSource] = Field(default_factory=list)
    # """Initial context sources to load"""

    config_file_path: str | None = None
    """Config file path for resolving environment."""

    knowledge: Knowledge | None = None
    """Knowledge sources for this agent."""

    workers: list[WorkerConfig] = Field(default_factory=list)
    """Worker agents which will be available as tools."""

    requires_tool_confirmation: ToolConfirmationMode = "per_tool"
    """How to handle tool confirmation:
    - "always": Always require confirmation for all tools
    - "never": Never require confirmation (ignore tool settings)
    - "per_tool": Use individual tool settings
    """

    debug: bool = False
    """Enable debug output for this agent."""

    def is_structured(self) -> bool:
        """Check if this config defines a structured agent."""
        return self.result_type is not None

    @model_validator(mode="before")
    @classmethod
    def validate_result_type(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Convert result type and apply its settings."""
        result_type = data.get("result_type")
        if isinstance(result_type, dict):
            # Extract response-specific settings
            tool_name = result_type.pop("result_tool_name", None)
            tool_description = result_type.pop("result_tool_description", None)
            retries = result_type.pop("result_retries", None)

            # Convert remaining dict to ResponseDefinition
            if "type" not in result_type:
                result_type["type"] = "inline"
            data["result_type"] = InlineResponseDefinition(**result_type)

            # Apply extracted settings to agent config
            if tool_name:
                data["result_tool_name"] = tool_name
            if tool_description:
                data["result_tool_description"] = tool_description
            if retries is not None:
                data["result_retries"] = retries

        return data

    @model_validator(mode="before")
    @classmethod
    def handle_model_types(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Convert model inputs to appropriate format."""
        model = data.get("model")
        match model:
            case str():
                data["model"] = {"type": "string", "identifier": model}
        return data

    async def get_toolsets(self) -> list[ResourceProvider]:
        """Get all resource providers for this agent."""
        providers: list[ResourceProvider] = []

        # Add providers from toolsets
        for toolset_config in self.toolsets:
            try:
                provider = toolset_config.get_provider()
                providers.append(provider)
            except Exception as e:
                logger.exception(
                    "Failed to create provider for toolset: %r", toolset_config
                )
                msg = f"Failed to create provider for toolset: {e}"
                raise ValueError(msg) from e

        return providers

    def get_tool_provider(self) -> ResourceProvider | None:
        """Get tool provider for this agent."""
        from llmling_agent.tools.base import Tool

        # Create provider for static tools
        if not self.tools:
            return None
        static_tools: list[Tool] = []
        for tool_config in self.tools:
            try:
                match tool_config:
                    case str():
                        if tool_config.startswith("crewai_tools"):
                            obj = import_class(tool_config)()
                            static_tools.append(Tool.from_crewai_tool(obj))
                        elif tool_config.startswith("langchain"):
                            obj = import_class(tool_config)()
                            static_tools.append(Tool.from_langchain_tool(obj))
                        else:
                            tool = Tool.from_callable(tool_config)
                            static_tools.append(tool)
                    case BaseToolConfig():
                        static_tools.append(tool_config.get_tool())
            except Exception:
                logger.exception("Failed to load tool %r", tool_config)
                continue

        return StaticResourceProvider(name="builtin", tools=static_tools)

    def get_session_config(self) -> MemoryConfig:
        """Get resolved memory configuration."""
        match self.session:
            case str() | UUID():
                return MemoryConfig(session=SessionQuery(name=str(self.session)))
            case SessionQuery():
                return MemoryConfig(session=self.session)
            case MemoryConfig():
                return self.session
            case None:
                return MemoryConfig()

    def get_system_prompts(self) -> list[BasePrompt]:
        """Get all system prompts as BasePrompts."""
        prompts: list[BasePrompt] = []
        for prompt in self.system_prompts:
            match prompt:
                case str():
                    # Convert string to StaticPrompt
                    static_prompt = StaticPrompt(
                        name="system",
                        description="System prompt",
                        messages=[PromptMessage(role="system", content=prompt)],
                    )
                    prompts.append(static_prompt)
                case BasePrompt():
                    prompts.append(prompt)
        return prompts

    def get_provider(self) -> AgentProvider:
        """Get resolved provider instance.

        Creates provider instance based on configuration:
        - Full provider config: Use as-is
        - Shorthand type: Create default provider config
        """
        # If string shorthand is used, convert to default provider config
        from llmling_agent_config.providers import (
            CallbackProviderConfig,
            HumanProviderConfig,
            LiteLLMProviderConfig,
            PydanticAIProviderConfig,
        )

        provider_config = self.provider
        if isinstance(provider_config, str):
            match provider_config:
                case "pydantic_ai":
                    provider_config = PydanticAIProviderConfig(model=self.model)
                case "human":
                    provider_config = HumanProviderConfig()
                case "litellm":
                    provider_config = LiteLLMProviderConfig(
                        model=self.model if isinstance(self.model, str) else None
                    )
                case _:
                    try:
                        fn = import_callable(provider_config)
                        provider_config = CallbackProviderConfig(fn=fn)
                    except Exception:  # noqa: BLE001
                        msg = f"Invalid provider type: {provider_config}"
                        raise ValueError(msg)  # noqa: B904

        # Create provider instance from config
        return provider_config.get_provider()

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

    @staticmethod
    def _resolve_environment_path(env: str, config_file_path: str | None = None) -> str:
        """Resolve environment path from config store or relative path."""
        from upath import UPath

        try:
            config_store = ConfigStore()
            return config_store.get_config(env)
        except KeyError:
            if config_file_path:
                base_dir = UPath(config_file_path).parent
                return str(base_dir / env)
            return env


if __name__ == "__main__":
    model = "openai:gpt-4o-mini"
    agent_cfg = AgentConfig(
        name="test_agent", model=model, tools=["crewai_tools.BraveSearchTool"]
    )  # type: ignore
    print(agent_cfg)
