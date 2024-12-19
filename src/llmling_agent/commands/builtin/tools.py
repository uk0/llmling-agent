"""Tool management commands."""

from __future__ import annotations

from typing import Any

from llmling.tools import LLMCallableTool
from llmling.utils.importing import import_callable

from llmling_agent.commands.base import Command, CommandContext
from llmling_agent.commands.exceptions import CommandError
from llmling_agent.log import get_logger


logger = get_logger(__name__)

CODE_TEMPLATE = '''\
def my_tool(text: str) -> str:
    """A new tool.

    Args:
        text: Input text

    Returns:
        Tool result
    """
    return f"You said: {text}"
'''


async def list_tools(
    ctx: CommandContext,
    args: list[str],
    kwargs: dict[str, str],
) -> None:
    """List all available tools."""
    agent = ctx.session._agent
    # Format output using ToolInfo formatting
    sections = ["# Available Tools\n"]
    for tool_info in agent.tools.values():
        status = "✓" if tool_info.enabled else "✗"
        sections.append(f"{status} {tool_info.format_info()}")

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

    tool_name = args[0]
    agent = ctx.session._agent

    try:
        tool_info = agent.tools[tool_name]

        # Start with the standard tool info format
        sections = [tool_info.format_info(indent="")]

        # Add extra metadata section if we have any additional info
        extra_info = []
        if tool_info.requires_capability:
            extra_info.append(f"Required Capability: {tool_info.requires_capability}")
        if tool_info.requires_confirmation:
            extra_info.append("Requires Confirmation: Yes")
        if tool_info.source != "runtime":  # Only show if not default
            extra_info.append(f"Source: {tool_info.source}")
        if tool_info.priority != 100:  # Only show if not default  # noqa: PLR2004
            extra_info.append(f"Priority: {tool_info.priority}")
        if tool_info.metadata:
            extra_info.append("\nMetadata:")
            extra_info.extend(f"- {k}: {v}" for k, v in tool_info.metadata.items())

        if extra_info:
            sections.append("\nAdditional Information:")
            sections.extend(extra_info)

        await ctx.output.print("\n".join(sections))
    except KeyError:
        await ctx.output.print(f"Tool '{tool_name}' not found")


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
            ctx.session._agent.tools.enable_tool(name)
        else:
            ctx.session._agent.tools.disable_tool(name)
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


async def register_tool(
    ctx: CommandContext,
    args: list[str],
    kwargs: dict[str, str],
) -> None:
    """Register a new tool from import path or function."""
    if not args:
        msg = "Usage: /register-tool <import_path> [--name name] [--description desc]"
        await ctx.output.print(msg)
        return

    import_path = args[0]
    name = kwargs.get("name")
    description = kwargs.get("description")

    try:
        callable_func = import_callable(import_path)

        # Create LLMCallableTool with optional overrides
        llm_tool = LLMCallableTool.from_callable(
            callable_func,
            name_override=name,
            description_override=description,
        )

        # Register with ToolManager
        meta = {"import_path": import_path, "registered_via": "register-tool"}
        tool_info = ctx.session._agent.tools.register_tool(
            llm_tool,
            enabled=True,
            source="dynamic",
            metadata=meta,
        )

        # Show the registered tool info
        await ctx.output.print("Tool registered successfully:")
        await ctx.output.print(tool_info.format_info())

    except Exception as e:
        msg = f"Failed to register tool: {e}"
        raise CommandError(msg) from e


async def write_tool(
    ctx: CommandContext,
    args: list[str],
    kwargs: dict[str, str],
) -> None:
    """Write and register a new tool interactively."""
    from prompt_toolkit import PromptSession
    from prompt_toolkit.lexers import PygmentsLexer
    from prompt_toolkit.styles import style_from_pygments_cls
    from pygments.lexers.python import PythonLexer
    from pygments.styles import get_style_by_name

    # Create editing session with syntax highlighting
    session: PromptSession[str] = PromptSession(
        lexer=PygmentsLexer(PythonLexer),
        multiline=True,
        style=style_from_pygments_cls(get_style_by_name("monokai")),
        include_default_pygments_style=False,
        mouse_support=True,
    )
    msg = "\nEnter tool code (ESC + Enter or Alt + Enter to save):\n\n"
    code = await session.prompt_async(msg, default=CODE_TEMPLATE)
    try:
        # Execute code in a namespace
        namespace: dict[str, Any] = {}
        exec(code, namespace)

        # Find all callable non-private functions
        tools = [
            v
            for v in namespace.values()
            if callable(v)
            and not v.__name__.startswith("_")
            and v.__code__.co_filename == "<string>"
        ]

        if not tools:
            await ctx.output.print("No tools found in code")
            return

        # Register all tools with ctx parameter added
        for func in tools:
            tool_info = ctx.session._agent.tools.register_tool(
                func, source="dynamic", metadata={"created_by": "write-tool"}
            )
            await ctx.output.print(f"Tool '{tool_info.name}' registered!")
            await ctx.output.print(tool_info.format_info())

    except Exception as e:
        msg = f"Error creating tools: {e}"
        raise CommandError(msg) from e


write_tool_cmd = Command(
    name="write-tool",
    description="Write and register new tools interactively",
    execute_func=write_tool,
    help_text=(
        "Opens an interactive Python editor to create new tools.\n"
        "- ESC + Enter or Alt + Enter to save and exit\n"
        "- Functions will be available as tools immediately\n\n"
        "Example template:\n"
        "def my_tool(text: str) -> str:\n"
        "    '''A new tool'''\n"
        "    return f'You said: {text}'\n"
    ),
    category="tools",
)

register_tool_cmd = Command(
    name="register-tool",
    description="Register a new tool from an import path",
    execute_func=register_tool,
    usage="<import_path> [--name name] [--description desc]",
    help_text=(
        "Register a new tool from a Python import path.\n"
        "Examples:\n"
        "  /register-tool webbrowser.open\n"
        "  /register-tool json.dumps --name format_json\n"
        "  /register-tool os.getcwd --description 'Get current directory'"
    ),
    category="tools",
)

list_tools_cmd = Command(
    name="list-tools",
    description="List all available tools",
    execute_func=list_tools,
    usage="[--source runtime|agent|builtin]",
    help_text=(
        "Show all available tools and their current status.\n"
        "Tools are grouped by source (runtime/agent/builtin).\n"
        "✓ indicates enabled, ✗ indicates disabled."
    ),
    category="tools",
)

tool_info_cmd = Command(
    name="show-tool",
    description="Show detailed information about a tool",
    execute_func=tool_info,
    usage="<name>",
    help_text=(
        "Display detailed information about a specific tool:\n"
        "- Source (runtime/agent/builtin)\n"
        "- Current status (enabled/disabled)\n"
        "- Priority and capabilities\n"
        "- Parameter descriptions\n"
        "- Additional metadata\n\n"
        "Example: /tool-info open_browser"
    ),
    category="tools",
)

enable_tool_cmd = Command(
    name="enable-tool",
    description="Enable a specific tool",
    execute_func=enable_tool,
    usage="<name>",
    help_text=(
        "Enable a previously disabled tool.\n"
        "Use /list-tools to see available tools.\n\n"
        "Example: /enable-tool open_browser"
    ),
    category="tools",
)

disable_tool_cmd = Command(
    name="disable-tool",
    description="Disable a specific tool",
    execute_func=disable_tool,
    usage="<name>",
    help_text=(
        "Disable a tool to prevent its use.\n"
        "Use /list-tools to see available tools.\n\n"
        "Example: /disable-tool open_browser"
    ),
    category="tools",
)
