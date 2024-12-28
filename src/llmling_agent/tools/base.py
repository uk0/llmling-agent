"""Base tool classes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from py2openai import OpenAIFunctionTool  # noqa: TC002

from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from llmling.tools import LLMCallableTool
    from py2openai.typedefs import ToolParameters
    from pydantic_ai import RunContext

    from llmling_agent.models import AgentContext

T = TypeVar("T")

logger = get_logger(__name__)


@dataclass(frozen=True)
class ToolContext:
    """Context for tool execution confirmation."""

    name: str
    """Name of the tool being executed"""

    args: dict[str, Any]
    """Arguments being passed to the tool"""

    schema: OpenAIFunctionTool
    """Complete OpenAI function schema"""

    runtime_ctx: RunContext[AgentContext]
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
class ToolInfo:
    """Information about a registered tool."""

    callable: LLMCallableTool
    """The actual tool implementation"""

    enabled: bool = True
    """Whether the tool is currently enabled"""

    source: Literal["runtime", "agent", "builtin", "dynamic"] = "runtime"
    """Where the tool came from."""

    priority: int = 100
    """Priority for tool execution (lower = higher priority)"""

    requires_confirmation: bool = False
    """Whether tool execution needs explicit confirmation"""

    requires_capability: str | None = None
    """Optional capability required to use this tool"""

    metadata: dict[str, str] = field(default_factory=dict)
    """Additional tool metadata"""

    @property
    def schema(self) -> OpenAIFunctionTool:
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
        properties: dict[str, Any] = schema.get("properties", {})  # type: ignore
        required: list[str] = schema.get("required", [])  # type: ignore

        params = []
        for name, details in properties.items():
            param = ToolParameter(
                name=name,
                required=name in required,
                type_info=details.get("type"),
                description=details.get("description"),
            )
            params.append(param)
        return params

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
            lines.extend(
                f"{indent}    {key}: {value}" for key, value in self.metadata.items()
            )
        return "\n".join(lines)


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
