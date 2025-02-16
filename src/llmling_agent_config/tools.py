"""Models for tools."""

from __future__ import annotations

from collections.abc import Callable  # noqa: TC003
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ImportString


if TYPE_CHECKING:
    from llmling_agent.tools.base import Tool


class BaseToolConfig(BaseModel):
    """Base configuration for agent tools."""

    type: str = Field(init=False)
    """Type discriminator for tool configs."""

    name: str | None = None
    """Optional override for the tool name."""

    description: str | None = None
    """Optional override for the tool description."""

    enabled: bool = True
    """Whether this tool is initially enabled."""

    requires_confirmation: bool = False
    """Whether tool execution needs confirmation."""

    requires_capability: str | None = None
    """Optional capability needed to use the tool."""

    priority: int = 100
    """Execution priority (lower = higher priority)."""

    cache_enabled: bool = False
    """Whether to enable result caching."""

    metadata: dict[str, str] = Field(default_factory=dict)
    """Additional tool metadata."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)

    def get_tool(self) -> Tool:
        """Convert config to Tool instance."""
        raise NotImplementedError


class ImportToolConfig(BaseToolConfig):
    """Configuration for importing tools from Python modules."""

    type: Literal["import"] = Field("import", init=False)
    """Import path based tool."""

    import_path: ImportString[Callable[..., Any]]
    """Import path to the tool function."""

    def get_tool(self) -> Tool:
        """Import and create tool from configuration."""
        from llmling_agent.tools.base import Tool

        return Tool.from_callable(
            self.import_path,
            name_override=self.name,
            description_override=self.description,
            enabled=self.enabled,
            requires_confirmation=self.requires_confirmation,
            requires_capability=self.requires_capability,
            priority=self.priority,
            cache_enabled=self.cache_enabled,
            metadata=self.metadata,
        )


class CrewAIToolConfig(BaseToolConfig):
    """Configuration for CrewAI-based tools."""

    type: Literal["crewai"] = Field("crewai", init=False)
    """CrewAI tool configuration."""

    import_path: ImportString
    """Import path to CrewAI tool class."""

    params: dict[str, Any] = Field(default_factory=dict)
    """Tool-specific parameters."""

    def get_tool(self) -> Tool:
        """Import and create CrewAI tool."""
        from llmling_agent.tools.base import Tool

        try:
            return Tool.from_crewai_tool(
                self.import_path(**self.params),
                name_override=self.name,
                description_override=self.description,
                enabled=self.enabled,
                requires_confirmation=self.requires_confirmation,
                requires_capability=self.requires_capability,
                priority=self.priority,
                cache_enabled=self.cache_enabled,
                metadata={"type": "crewai", **self.metadata},
            )
        except ImportError as e:
            msg = "CrewAI not installed. Install with: pip install crewai-tools"
            raise ImportError(msg) from e


class LangChainToolConfig(BaseToolConfig):
    """Configuration for LangChain tools."""

    type: Literal["langchain"] = Field("langchain", init=False)
    """LangChain tool configuration."""

    tool_name: str
    """Name of LangChain tool to use."""

    params: dict[str, Any] = Field(default_factory=dict)
    """Tool-specific parameters."""

    def get_tool(self) -> Tool:
        """Import and create LangChain tool."""
        try:
            from langchain.tools import load_tool

            from llmling_agent.tools.base import Tool

            return Tool.from_langchain_tool(
                load_tool(self.tool_name, **self.params),
                name_override=self.name,
                description_override=self.description,
                enabled=self.enabled,
                requires_confirmation=self.requires_confirmation,
                requires_capability=self.requires_capability,
                priority=self.priority,
                cache_enabled=self.cache_enabled,
                metadata={"type": "langchain", **self.metadata},
            )
        except ImportError as e:
            msg = "LangChain not installed. Install with: pip install langchain"
            raise ImportError(msg) from e


# Union type for tool configs
ToolConfig = Annotated[
    ImportToolConfig | CrewAIToolConfig | LangChainToolConfig,
    Field(discriminator="type"),
]
