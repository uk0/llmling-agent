"""Request information formatting for agent interactions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from llmling_agent.pydantic_ai_utils import format_response


if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic_ai import Tool
    from pydantic_ai._result import ResultSchema
    from pydantic_ai.messages import Message


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


@dataclass
class ToolInfo:
    """Information about an available tool."""

    name: str
    parameters: list[ToolParameter]
    description: str | None = None

    def format(self, indent: str = "  ") -> str:
        """Format tool information."""
        lines = [f"{indent}→ {self.name}"]
        if self.description:
            lines.append(f"{indent}  {self.description}")
        if self.parameters:
            lines.append(f"{indent}  Parameters:")
            lines.extend(f"{indent}    {param}" for param in self.parameters)
        return "\n".join(lines)


def extract_tool_info(tool: Tool[Any]) -> ToolInfo:
    """Extract tool information from pydantic-ai Tool."""
    schema = tool._parameters_json_schema
    properties = schema.get("properties", {})
    required = schema.get("required", [])

    parameters = []
    for name, details in properties.items():
        param = ToolParameter(
            name=name,
            required=name in required,
            type_info=details.get("type"),
            description=details.get("description"),
        )
        parameters.append(param)

    return ToolInfo(name=tool.name, description=tool.description, parameters=parameters)


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


def format_request_info(
    prompt: str,
    tools: list[Tool[Any]],
    new_messages: Sequence[Message],
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
        sections.extend([
            "",
            "Available Tools",
            "-" * 15,
        ])
        for tool in tools:
            tool_info = extract_tool_info(tool)
            sections.append(tool_info.format())

    # Message information
    if new_messages:
        sections.extend([
            "",
            "New Context Messages",
            "-" * 18,
        ])
        for msg in new_messages:
            content = format_response(msg)
            sections.append(f"  [{msg.role}] {content[:100]}...")

    # Current prompt
    sections.extend([
        "",
        "Current Prompt",
        "-" * 13,
        prompt,
    ])

    return "\n".join(sections)
