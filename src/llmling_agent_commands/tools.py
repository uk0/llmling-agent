"""Tool management commands."""

from __future__ import annotations

from typing import Any

from llmling.utils.importing import import_callable
from slashed import Command, CommandContext, CommandError, CompletionContext
from slashed.completers import CallbackCompleter

from llmling_agent.agent.context import AgentContext  # noqa: TC001
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

TOOL_INFO_HELP = """\
Display detailed information about a specific tool:
- Source (runtime/agent/builtin)
- Current status (enabled/disabled)
- Priority and capabilities
- Parameter descriptions
- Additional metadata

Example: /tool-info open_browser
"""

WRITE_TOOL_HELP = """\
Opens an interactive Python editor to create new tools.
- ESC + Enter or Alt + Enter to save and exit
- Functions will be available as tools immediately

Example template:
def my_tool(text: str) -> str:
    '''A new tool'''
    return f'You said: {text}'
"""

REGISTER_TOOL_HELP = """\
Register a new tool from a Python import path.
Examples:
  /register-tool webbrowser.open
  /register-tool json.dumps --name format_json
  /register-tool os.getcwd --description 'Get current directory'
"""

ENABLE_TOOL_HELP = """\
Enable a previously disabled tool.
Use /list-tools to see available tools.

Example: /enable-tool open_browser
"""

DISABLE_TOOL_HELP = """\
Disable a tool to prevent its use.
Use /list-tools to see available tools.

Example: /disable-tool open_browser
"""

LIST_TOOLS_HELP = """\
Show all available tools and their current status.
Tools are grouped by source (runtime/agent/builtin).
✓ indicates enabled, ✗ indicates disabled.
"""


async def list_tools(
    ctx: CommandContext[AgentContext],
    args: list[str],
    kwargs: dict[str, str],
):
    """List all available tools."""
    agent = ctx.context.agent
    # Format output using Tool formatting
    sections = ["# Available Tools\n"]
    for tool_info in agent.tools.values():
        status = "✓" if tool_info.enabled else "✗"
        sections.append(f"{status} {tool_info.format_info()}")

    await ctx.output.print("\n".join(sections))


async def tool_info(
    ctx: CommandContext[AgentContext],
    args: list[str],
    kwargs: dict[str, str],
):
    """Show detailed information about a tool."""
    if not args:
        await ctx.output.print("Usage: /show-tool <name>")
        return

    tool_name = args[0]
    agent = ctx.context.agent

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
    ctx: CommandContext[AgentContext],
    args: list[str],
    kwargs: dict[str, str],
    *,
    enable: bool,
):
    """Enable or disable a tool."""
    if not args:
        action = "enable" if enable else "disable"
        await ctx.output.print(f"Usage: /{action}-tool <name>")
        return

    name = args[0]
    try:
        if enable:
            ctx.context.agent.tools.enable_tool(name)
        else:
            ctx.context.agent.tools.disable_tool(name)
        action = "enabled" if enable else "disabled"
        await ctx.output.print(f"Tool '{name}' {action}")
    except ValueError as e:
        msg = f"Failed to {'enable' if enable else 'disable'} tool: {e}"
        raise CommandError(msg) from e


async def enable_tool(
    ctx: CommandContext[AgentContext],
    args: list[str],
    kwargs: dict[str, str],
):
    """Enable a tool."""
    await toggle_tool(ctx, args, kwargs, enable=True)


async def disable_tool(
    ctx: CommandContext[AgentContext],
    args: list[str],
    kwargs: dict[str, str],
):
    """Disable a tool."""
    await toggle_tool(ctx, args, kwargs, enable=False)


async def register_tool(
    ctx: CommandContext[AgentContext],
    args: list[str],
    kwargs: dict[str, str],
):
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
        # Register with ToolManager
        tool_info = ctx.context.agent.tools.register_tool(
            callable_func,
            name_override=name,
            description_override=description,
            enabled=True,
            source="dynamic",
            metadata={"import_path": import_path, "registered_via": "register-tool"},
        )

        # Show the registered tool info
        info = tool_info.format_info()
        await ctx.output.print(f"Tool registered successfully:\n {info}")

    except Exception as e:
        msg = f"Failed to register tool: {e}"
        raise CommandError(msg) from e


async def write_tool(
    ctx: CommandContext[AgentContext],
    args: list[str],
    kwargs: dict[str, str],
):
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
            tool_info = ctx.context.agent.tools.register_tool(
                func, source="dynamic", metadata={"created_by": "write-tool"}
            )
            info = tool_info.format_info()
            await ctx.output.print(f"Tool '{tool_info.name}' registered!\n{info}")

    except Exception as e:
        msg = f"Error creating tools: {e}"
        raise CommandError(msg) from e


async def get_tool_names(ctx: CompletionContext[AgentContext]) -> list[str]:
    return list(await ctx.command_context.context.agent.tools.get_tool_names())


write_tool_cmd = Command(
    name="write-tool",
    description="Write and register new tools interactively",
    execute_func=write_tool,
    help_text=WRITE_TOOL_HELP,
    category="tools",
)

register_tool_cmd = Command(
    name="register-tool",
    description="Register a new tool from an import path",
    execute_func=register_tool,
    usage="<import_path> [--name name] [--description desc]",
    help_text=REGISTER_TOOL_HELP,
    category="tools",
)

list_tools_cmd = Command(
    name="list-tools",
    description="List all available tools",
    execute_func=list_tools,
    usage="[--source runtime|agent|builtin]",
    help_text=LIST_TOOLS_HELP,
    category="tools",
)

tool_info_cmd = Command(
    name="show-tool",
    description="Show detailed information about a tool",
    execute_func=tool_info,
    usage="<name>",
    help_text=TOOL_INFO_HELP,
    category="tools",
    completer=CallbackCompleter(get_tool_names),
)

enable_tool_cmd = Command(
    name="enable-tool",
    description="Enable a specific tool",
    execute_func=enable_tool,
    usage="<name>",
    help_text=ENABLE_TOOL_HELP,
    category="tools",
    completer=CallbackCompleter(get_tool_names),
)

disable_tool_cmd = Command(
    name="disable-tool",
    description="Disable a specific tool",
    execute_func=disable_tool,
    usage="<name>",
    help_text=DISABLE_TOOL_HELP,
    category="tools",
    completer=CallbackCompleter(get_tool_names),
)
