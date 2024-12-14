from __future__ import annotations

from dataclasses import dataclass
import inspect
from inspect import Parameter, Signature
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from py2openai import OpenAIFunctionTool  # noqa: TC002
from pydantic_ai import RunContext, Tool

from llmling_agent.context import AgentContext
from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from py2openai.typedefs import ToolParameters

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


class ToolConfirmation(Protocol):
    """Protocol for tool confirmation handlers."""

    async def confirm_tool(self, context: ToolContext) -> bool:
        """Request confirmation for tool execution.

        Args:
            context: Complete context about the tool execution

        Returns:
            Whether the tool execution was confirmed
        """
        ...


class ToolExecutionDeniedError(Exception):
    """Raised when tool execution is denied by user."""


def create_confirmed_tool_wrapper(
    name: str,
    schema: OpenAIFunctionTool,
    original_callable: Callable[..., T | Awaitable[T]] | None = None,
    *,
    confirm_callback: Callable[[ToolContext], Awaitable[bool]] | None = None,
) -> Callable[..., Awaitable[T]]:
    """Create a tool wrapper function with confirmation support."""
    # Create base signature
    if original_callable:
        sig = inspect.signature(original_callable)
        params = [
            Parameter(
                "ctx",
                Parameter.POSITIONAL_OR_KEYWORD,
                annotation=RunContext[AgentContext],
            ),
            *[
                Parameter(name, p.kind, annotation=p.annotation, default=p.default)
                for name, p in sig.parameters.items()
            ],
        ]
        return_annotation = sig.return_annotation
    else:
        params = [
            Parameter(
                "ctx",
                Parameter.POSITIONAL_OR_KEYWORD,
                annotation=RunContext[AgentContext],
            )
        ]
        properties = schema["function"].get("parameters", {}).get("properties", {})
        for prop_name, info in properties.items():
            default = Parameter.empty if info.get("required") else None
            param = Parameter(
                prop_name,
                Parameter.POSITIONAL_OR_KEYWORD,
                annotation=Any,
                default=default,
            )
            params.append(param)
        return_annotation = Any

    # Create function signature
    sig = Signature(params, return_annotation=return_annotation)

    async def confirmed_tool_wrapper(*args: Any, **kwargs: Any) -> Any:
        ctx = args[0]  # First arg is always context

        # Handle confirmation if needed
        if confirm_callback:
            tool_ctx = ToolContext(
                name=name,
                args=kwargs,
                schema=schema,
                runtime_ctx=ctx,
            )
            confirmed = await confirm_callback(tool_ctx)
            if not confirmed:
                msg = f"User denied execution of {name}"
                raise ToolExecutionDeniedError(msg)

        return await ctx.deps.execute_tool(name, **kwargs)

    # Apply signature and metadata
    confirmed_tool_wrapper.__signature__ = sig  # type: ignore
    confirmed_tool_wrapper.__name__ = f"confirmed_{schema['function']['name']}"
    confirmed_tool_wrapper.__doc__ = schema["function"]["description"]
    confirmed_tool_wrapper.__annotations__ = {p.name: p.annotation for p in params}

    return confirmed_tool_wrapper


def create_runtime_tool_wrapper(
    name: str,
    schema: OpenAIFunctionTool,
    description: str | None = None,
) -> Tool[AgentContext]:
    """Create wrapper for a runtime tool.

    Args:
        name: Name of the tool
        schema: OpenAI function schema for the tool
        description: Optional tool description

    Returns:
        Tool instance with proper wrapper function
    """
    # Create function signature based on schema
    params = [
        Parameter(
            "ctx",
            Parameter.POSITIONAL_OR_KEYWORD,
            annotation=RunContext[AgentContext],
        )
    ]

    # Add parameters from schema
    properties = schema["function"].get("parameters", {}).get("properties", {})
    for prop_name, info in properties.items():
        default = Parameter.empty if info.get("required") else None
        param = Parameter(
            prop_name,
            Parameter.POSITIONAL_OR_KEYWORD,
            annotation=Any,
            default=default,
        )
        params.append(param)

    # Create signature
    sig = Signature(params, return_annotation=Any)

    async def tool_wrapper(*args: Any, **kwargs: Any) -> Any:
        ctx = args[0]  # First arg is always context
        if not ctx.deps.runtime:
            msg = f"No runtime available for tool {name}"
            raise RuntimeError(msg)
        logger.debug("Executing tool %s with args: %s", name, kwargs)
        return await ctx.deps.runtime.execute_tool(name, **kwargs)

    # Apply signature and metadata
    tool_wrapper.__signature__ = sig  # type: ignore
    tool_wrapper.__name__ = name
    tool_wrapper.__doc__ = schema["function"]["description"]
    tool_wrapper.__annotations__ = {p.name: p.annotation for p in params}

    return Tool(
        tool_wrapper,
        takes_ctx=True,
        name=name,
        description=description,
    )
