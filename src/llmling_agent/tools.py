from __future__ import annotations

import inspect
from inspect import Parameter, Signature
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from llmling.config.runtime import RuntimeConfig
from pydantic_ai import RunContext


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from py2openai import OpenAIFunctionTool


POS_OR_KEY = Parameter.POSITIONAL_OR_KEYWORD
T = TypeVar("T")


class ToolConfirmation(Protocol):
    """Protocol for tool confirmation handlers."""

    async def confirm_tool(
        self,
        tool_name: str,
        args: dict[str, Any],
        description: str | None = None,
    ) -> bool:
        """Request confirmation for tool execution."""


def create_tool_wrapper(
    name: str,
    schema: OpenAIFunctionTool,
    original_callable: Callable[..., T | Awaitable[T]] | None = None,
) -> Callable[..., Awaitable[T]]:
    """Create a tool wrapper function with proper signature and type hints.

    Creates an async wrapper function that forwards calls to RuntimeConfig.execute_tool.
    If the original callable is provided, its signature and type hints are preserved.
    Otherwise, the signature is reconstructed from the OpenAI function schema.

    Args:
        name: Name of the tool to wrap
        schema: OpenAI function schema (from py2openai)
        original_callable: Optional original function to preserve signature from

    Returns:
        Async wrapper function with proper signature that delegates to execute_tool
    """
    # If we have the original callable, use its signature
    if original_callable:
        # Create parameters with original types
        sig = inspect.signature(original_callable)
        params = [
            Parameter("ctx", POS_OR_KEY, annotation=RunContext[RuntimeConfig]),
            *[
                Parameter(name, p.kind, annotation=p.annotation, default=p.default)
                for name, p in sig.parameters.items()
            ],
        ]
        return_annotation = sig.return_annotation
    else:
        # Fall back to schema-based parameters with Any types
        params = [Parameter("ctx", POS_OR_KEY, annotation=RunContext[RuntimeConfig])]
        properties = schema["function"].get("parameters", {}).get("properties", {})
        for prop_name, info in properties.items():
            default = Parameter.empty if info.get("required") else None
            param = Parameter(prop_name, POS_OR_KEY, annotation=Any, default=default)
            params.append(param)
        return_annotation = Any

    # Create the signature
    sig = Signature(params, return_annotation=return_annotation)

    # Create the wrapper function
    async def tool_wrapper(*args: Any, **kwargs: Any) -> Any:
        ctx = args[0]  # First arg is always context
        return await ctx.deps.execute_tool(name, **kwargs)

    # Apply the signature and metadata
    tool_wrapper.__signature__ = sig  # type: ignore
    tool_wrapper.__name__ = schema["function"]["name"]
    tool_wrapper.__doc__ = schema["function"]["description"]
    tool_wrapper.__annotations__ = {p.name: p.annotation for p in params}

    return tool_wrapper
