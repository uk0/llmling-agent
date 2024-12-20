"""Request information formatting for agent interactions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
)

from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from pydantic_ai._result import ResultSchema

    from llmling_agent.tools.base import ToolInfo


logger = get_logger(__name__)


def format_result_schema(schema: ResultSchema[Any] | None) -> str:
    """Format result schema information."""
    if not schema:
        return "str (free text)"

    parts = []
    if schema.allow_text_result:
        parts.append("Allows free text")

    for name, tool in schema.tools.items():
        params = tool.tool_def.parameters_json_schema["function"]["parameters"]
        type_info = params.get("type", "object")
        if "properties" in params:
            properties = params["properties"]
            fields = [
                f"    {field}: {info.get('type', 'any')}"
                for field, info in properties.items()
            ]
            type_info = "{\n" + "\n".join(fields) + "\n  }"
        parts.append(f"  {name}: {type_info}")

    return "\n  ".join(parts)


def format_messages(messages: list[ModelMessage], indent: str = "  ") -> list[str]:
    """Format model messages."""
    formatted = []
    for msg in messages:
        match msg:
            case ModelRequest() as req:
                for p in req.parts:
                    content = str(p.content)
                    formatted.append(f"{indent}[{p.part_kind}] {content[:100]}...")
            case ModelResponse() as resp:
                for part in resp.parts:
                    match part:
                        case TextPart():
                            content = part.content
                        case ToolCallPart():
                            args = (
                                part.args.args_dict  # pyright: ignore
                                if hasattr(part.args, "args_dict")
                                else part.args.args_json  # pyright: ignore
                            )
                            content = f"Tool: {part.tool_name}, Args: {args}"
                        case _:
                            content = str(part)
                    formatted.append(f"{indent}[{part.part_kind}] {content[:100]}...")
    return formatted


def format_request_info(
    prompt: str,
    tools: list[ToolInfo],
    new_messages: list[ModelMessage],
    model: str | None = None,
    result_schema: ResultSchema[Any] | None = None,
) -> str:
    """Format complete request information.

    Args:
        prompt: Current prompt being processed
        tools: Available pydantic-ai Tool instances
        new_messages: New messages being added to context
        model: Model being used (if specified)
        result_schema: Schema for expected results
    """
    sections = [
        "Request Information",
        "=" * 50,
        "",
        "Configuration",
        "-" * 12,
        f"Model: {model or 'default'}",
        "Expected Result:",
        "  " + format_result_schema(result_schema),
    ]

    # Tool information
    if tools:
        sections.extend(["", "Available Tools", "-" * 15])
        sections.extend(tool.format_info() for tool in tools)

    # Message information
    if new_messages:
        sections.extend(["", "New Context Messages", "-" * 18])
        sections.extend(format_messages(new_messages))

    # Current prompt
    sections.extend(["", "Current Prompt", "-" * 13, prompt])

    return "\n".join(sections)
