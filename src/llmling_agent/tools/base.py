"""Base tool classes."""

from __future__ import annotations

from dataclasses import dataclass, field
import inspect
from typing import TYPE_CHECKING, Any, Literal, Self, TypeVar

from llmling import LLMCallableTool
import py2openai  # noqa: TC002

from llmling_agent.log import get_logger
from llmling_agent.observability import track_tool
from llmling_agent.utils.inspection import execute


if TYPE_CHECKING:
    from collections.abc import Callable

    from py2openai.typedefs import Property, ToolParameters

    from llmling_agent.agent import AgentContext
    from llmling_agent.common_types import ToolSource

T = TypeVar("T")

logger = get_logger(__name__)


@dataclass(frozen=True)
class ToolContext:
    """Context for tool execution confirmation."""

    name: str
    """Name of the tool being executed"""

    args: dict[str, Any]
    """Arguments being passed to the tool"""

    schema: py2openai.OpenAIFunctionTool
    """Complete OpenAI function schema"""

    runtime_ctx: AgentContext
    """Runtime context from agent"""

    @property
    def description(self) -> str | None:
        """Get tool description from schema."""
        return self.schema["function"].get("description")

    @property
    def parameters(self) -> ToolParameters:
        """Get parameter definitions from schema."""
        return self.schema["function"].get("parameters", {})  # type: ignore

    def __str__(self) -> str:
        """Format tool context for logging/display."""
        return (
            f"Tool: {self.name}\n"
            f"Arguments: {self.args}\n"
            f"Description: {self.description or 'N/A'}"
        )


@dataclass
class Tool:
    """Information about a registered tool."""

    callable: LLMCallableTool
    """The actual tool implementation"""

    enabled: bool = True
    """Whether the tool is currently enabled"""

    source: ToolSource = "runtime"
    """Where the tool came from."""

    priority: int = 100
    """Priority for tool execution (lower = higher priority)"""

    requires_confirmation: bool = False
    """Whether tool execution needs explicit confirmation"""

    requires_capability: str | None = None
    """Optional capability required to use this tool"""

    agent_name: str | None = None
    """The agent name as an identifier for agent-as-a-tool."""

    metadata: dict[str, str] = field(default_factory=dict)
    """Additional tool metadata"""

    cache_enabled: bool = False
    """Whether to enable caching for this tool."""

    @property
    def schema(self) -> py2openai.OpenAIFunctionTool:
        """Get the OpenAI function schema for the tool."""
        return self.callable.get_schema()

    @property
    def name(self) -> str:
        """Get tool name."""
        return self.callable.name

    @property
    def description(self) -> str | None:
        """Get tool description."""
        return self.callable.description

    def matches_filter(self, state: Literal["all", "enabled", "disabled"]) -> bool:
        """Check if tool matches state filter."""
        match state:
            case "all":
                return True
            case "enabled":
                return self.enabled
            case "disabled":
                return not self.enabled

    @property
    def parameters(self) -> list[ToolParameter]:
        """Get information about tool parameters."""
        schema = self.schema["function"]
        properties: dict[str, Property] = schema.get("properties", {})  # type: ignore
        required: list[str] = schema.get("required", [])  # type: ignore

        return [
            ToolParameter(
                name=name,
                required=name in required,
                type_info=details.get("type"),
                description=details.get("description"),
            )
            for name, details in properties.items()
        ]

    def format_info(self, indent: str = "  ") -> str:
        """Format complete tool information."""
        lines = [f"{indent}→ {self.name}"]
        if self.description:
            lines.append(f"{indent}  {self.description}")
        if self.parameters:
            lines.append(f"{indent}  Parameters:")
            lines.extend(f"{indent}    {param}" for param in self.parameters)
        if self.metadata:
            lines.append(f"{indent}  Metadata:")
            lines.extend(f"{indent}    {k}: {v}" for k, v in self.metadata.items())
        return "\n".join(lines)

    async def execute(self, *args: Any, **kwargs: Any) -> Any:
        """Execute tool, handling both sync and async cases."""
        fn = track_tool(self.name)(self.callable.callable)
        return await execute(fn, *args, **kwargs, use_thread=True)

    @classmethod
    def from_code(
        cls,
        code: str,
        name: str | None = None,
        description: str | None = None,
    ) -> Self:
        """Create a tool from a code string."""
        namespace: dict[str, Any] = {}
        exec(code, namespace)
        func = next((v for v in namespace.values() if callable(v)), None)
        if not func:
            msg = "No callable found in provided code"
            raise ValueError(msg)
        return cls.from_callable(
            func, name_override=name, description_override=description
        )

    @classmethod
    def from_callable(
        cls,
        fn: Callable[..., Any] | str,
        *,
        name_override: str | None = None,
        description_override: str | None = None,
        schema_override: py2openai.OpenAIFunctionDefinition | None = None,
        **kwargs: Any,
    ) -> Self:
        tool = LLMCallableTool.from_callable(
            fn,
            name_override=name_override,
            description_override=description_override,
            schema_override=schema_override,
        )
        return cls(tool, **kwargs)

    @classmethod
    def from_crewai_tool(
        cls,
        tool: Any,
        *,
        name_override: str | None = None,
        description_override: str | None = None,
        schema_override: py2openai.OpenAIFunctionDefinition | None = None,
        **kwargs: Any,
    ) -> Self:
        """Allows importing crewai tools."""
        # vaidate_import("crewai_tools", "crewai")
        try:
            from crewai.tools import BaseTool as CrewAiBaseTool
        except ImportError as e:
            msg = "crewai package not found. Please install it with 'pip install crewai'"
            raise ImportError(msg) from e

        if not isinstance(tool, CrewAiBaseTool):
            msg = f"Expected CrewAI BaseTool, got {type(tool)}"
            raise TypeError(msg)

        return cls.from_callable(
            tool._run,
            name_override=name_override or tool.__class__.__name__.removesuffix("Tool"),
            description_override=description_override or tool.description,
            schema_override=schema_override,
            **kwargs,
        )

    @classmethod
    def from_langchain_tool(
        cls,
        tool: Any,
        *,
        name_override: str | None = None,
        description_override: str | None = None,
        schema_override: py2openai.OpenAIFunctionDefinition | None = None,
        **kwargs: Any,
    ) -> Self:
        """Create a tool from a LangChain tool."""
        # vaidate_import("langchain_core", "langchain")
        try:
            from langchain_core.tools import BaseTool as LangChainBaseTool
        except ImportError as e:
            msg = "langchain-core package not found."
            raise ImportError(msg) from e

        if not isinstance(tool, LangChainBaseTool):
            msg = f"Expected LangChain BaseTool, got {type(tool)}"
            raise TypeError(msg)

        return cls.from_callable(
            tool.invoke,
            name_override=name_override or tool.name,
            description_override=description_override or tool.description,
            schema_override=schema_override,
            **kwargs,
        )

    @classmethod
    def from_autogen_tool(
        cls,
        tool: Any,
        *,
        name_override: str | None = None,
        description_override: str | None = None,
        schema_override: py2openai.OpenAIFunctionDefinition | None = None,
        **kwargs: Any,
    ) -> Self:
        """Create a tool from a AutoGen tool."""
        # vaidate_import("autogen_core", "autogen")
        try:
            from autogen_core import CancellationToken
            from autogen_core.tools import BaseTool
        except ImportError as e:
            msg = "autogent_core package not found."
            raise ImportError(msg) from e

        if not isinstance(tool, BaseTool):
            msg = f"Expected AutoGent BaseTool, got {type(tool)}"
            raise TypeError(msg)
        token = CancellationToken()

        input_model = tool.__class__.__orig_bases__[0].__args__[0]  # type: ignore

        name = name_override or tool.name or tool.__class__.__name__.removesuffix("Tool")
        description = (
            description_override
            or tool.description
            or inspect.getdoc(tool.__class__)
            or ""
        )

        async def wrapper(**kwargs: Any) -> Any:
            # Convert kwargs to the expected input model
            model = input_model(**kwargs)
            return await tool.run(model, cancellation_token=token)

        return cls.from_callable(
            wrapper,  # type: ignore
            name_override=name,
            description_override=description,
            schema_override=schema_override,
            **kwargs,
        )


@dataclass
class ToolParameter:
    """Information about a tool parameter."""

    name: str
    required: bool
    type_info: str | None = None
    description: str | None = None

    def __str__(self) -> str:
        """Format parameter info."""
        req = "*" if self.required else ""
        type_str = f": {self.type_info}" if self.type_info else ""
        desc = f" - {self.description}" if self.description else ""
        return f"{self.name}{req}{type_str}{desc}"
