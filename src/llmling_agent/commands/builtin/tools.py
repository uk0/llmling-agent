from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from llmling_agent.commands.base import Command, CommandContext, CommandError
from llmling_agent.log import get_logger


logger = get_logger(__name__)


@dataclass
class ToolInfo:
    """Information about a tool."""

    name: str
    description: str | None
    source: Literal["runtime", "agent", "builtin"]
    enabled: bool
    schema: dict | None = None


async def list_tools(
    ctx: CommandContext,
    args: list[str],
    kwargs: dict[str, str],
) -> None:
    """List all available tools."""
    agent = ctx.session._agent
    tool_states = ctx.session.get_tool_states()

    # Collect all tool info
    tools: list[ToolInfo] = []

    # Runtime tools
    for name in agent.runtime.tools:
        tool_def = agent.runtime.tools[name]
        info = ToolInfo(
            name=name,
            description=tool_def.description,
            source="runtime",
            enabled=tool_states.get(name, False),
            schema=dict(tool_def.get_schema()),
        )
        tools.append(info)

    # Agent's own tools
    for tool in agent._original_tools:
        info = ToolInfo(
            name=tool.name,
            description=tool.description,
            source="agent",
            enabled=tool_states.get(tool.name, False),
        )
        tools.append(info)

    # Format output
    sections = ["# Available Tools\n"]

    for source in ["runtime", "agent", "builtin"]:
        source_tools = [t for t in tools if t.source == source]
        if source_tools:
            sections.append(f"\n## {source.title()} Tools")
            for tool in source_tools:
                status = "✓" if tool.enabled else "✗"
                desc = f": {tool.description.split('\n')[0]}" if tool.description else ""
                sections.append(f"- {status} **{tool.name}**{desc}")

    await ctx.output.print("\n".join(sections))


async def tool_info(
    ctx: CommandContext,
    args: list[str],
    kwargs: dict[str, str],
) -> None:
    """Show detailed information about a tool."""
    if not args:
        await ctx.output.print("Usage: /tool-info <name>")
        return

    name = args[0]
    agent = ctx.session._agent
    tool_states = ctx.session.get_tool_states()

    # Check runtime tools
    if name in agent.runtime.tools:
        tool_def = agent.runtime.tools[name]
        schema = tool_def.get_schema()
        sections = [
            f"# Tool: {name}",
            "\n## Details",
            "- **Source**: Runtime",
            f"- **Enabled**: {'Yes' if tool_states.get(name, False) else 'No'}",
            f"- **Description**: {tool_def.description or 'N/A'}",
        ]
        if schema:
            sections.extend([
                "\n## Schema",
                "```json",
                str(schema),
                "```",
            ])
        await ctx.output.print("\n".join(sections))
        return

    # Check agent tools
    for tool in agent._original_tools:
        if tool.name == name:
            sections = [
                f"# Tool: {name}",
                "\n## Details",
                "- **Source**: Agent",
                f"- **Enabled**: {'Yes' if tool_states.get(name, False) else 'No'}",
                f"- **Description**: {tool.description or 'N/A'}",
            ]
            await ctx.output.print("\n".join(sections))
            return

    await ctx.output.print(f"Tool '{name}' not found")


async def toggle_tool(
    ctx: CommandContext,
    args: list[str],
    kwargs: dict[str, str],
    *,
    enable: bool,
) -> None:
    """Enable or disable a tool."""
    if not args:
        action = "enable" if enable else "disable"
        await ctx.output.print(f"Usage: /{action}-tool <name>")
        return

    name = args[0]
    try:
        if enable:
            ctx.session._agent.enable_tool(name)
        else:
            ctx.session._agent.disable_tool(name)
        action = "enabled" if enable else "disabled"
        await ctx.output.print(f"Tool '{name}' {action}")
    except ValueError as e:
        msg = f"Failed to {'enable' if enable else 'disable'} tool: {e}"
        raise CommandError(msg) from e


async def enable_tool(
    ctx: CommandContext,
    args: list[str],
    kwargs: dict[str, str],
) -> None:
    """Enable a tool."""
    await toggle_tool(ctx, args, kwargs, enable=True)


async def disable_tool(
    ctx: CommandContext,
    args: list[str],
    kwargs: dict[str, str],
) -> None:
    """Disable a tool."""
    await toggle_tool(ctx, args, kwargs, enable=False)


# Command definitions
list_tools_cmd = Command(
    name="list-tools",
    description="List all available tools",
    execute_func=list_tools,
    category="tools",
)

tool_info_cmd = Command(
    name="tool-info",
    description="Show detailed information about a tool",
    execute_func=tool_info,
    usage="<name>",
    category="tools",
)

enable_tool_cmd = Command(
    name="enable-tool",
    description="Enable a specific tool",
    execute_func=enable_tool,
    usage="<name>",
    category="tools",
)

disable_tool_cmd = Command(
    name="disable-tool",
    description="Disable a specific tool",
    execute_func=disable_tool,
    usage="<name>",
    category="tools",
)
