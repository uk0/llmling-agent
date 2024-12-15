from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from llmling_agent.commands.base import Command, CommandContext, CommandError
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
    tool_states = agent.list_tools()  # Single source of truth

    # Collect all tool info
    tools: list[ToolInfo] = []

    # Runtime tools
    for name in agent.runtime.tools:
        tool_def = agent.runtime.tools[name]
        info = ToolInfo(
            name=name,
            description=tool_def.description,
            source="runtime",
            enabled=tool_states[name],
            schema=dict(tool_def.get_schema()),
        )
        tools.append(info)

    # Agent's custom tools
    for tool in agent._original_tools:
        info = ToolInfo(
            name=tool.name,
            description=tool.description,
            source="agent",
            enabled=tool_states[tool.name],  # Use same tool_states dict
        )
        tools.append(info)

    # Format output
    sections = ["# Available Tools\n"]
    for source in ["runtime", "agent", "builtin"]:
        source_tools = [t for t in tools if t.source == source]
        if source_tools:
            sections.append(f"\n## {source.title()} Tools")
            for t in source_tools:
                status = "✓" if t.enabled else "✗"
                desc = f": {t.description.split('\n')[0]}" if t.description else ""
                sections.append(f"- {status} **{t.name}**{desc}")

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
        if not ctx.session._agent._context.runtime:
            msg = "No runtime available"
            raise RuntimeError(msg)  # noqa: TRY301

        result = await ctx.session._agent._context.runtime.register_tool(
            import_path,
            name=name,
            description=description,
            auto_install=True,
        )
        await ctx.output.print(result)
        # Enable the tool automatically
        tool_name = name if name else import_path.split(".")[-1]
        ctx.session._agent.enable_tool(tool_name)
        await ctx.output.print(f"Tool '{tool_name}' automatically enabled")

    except Exception as e:  # noqa: BLE001
        await ctx.output.print(f"Failed to register tool: {e}")


async def use_tool(
    ctx: CommandContext,
    args: list[str],
    kwargs: dict[str, str],
) -> None:
    """Execute prompt with a specific tool."""
    if len(args) < 2:  # noqa: PLR2004
        await ctx.output.print("Usage: /use-tool <tool_name> <prompt>")
        return

    tool_name = args[0]
    prompt = " ".join(args[1:])  # everything after tool name is the prompt
    agent = ctx.session._agent

    if tool_name not in agent.runtime.tools and not any(
        t.name == tool_name for t in agent._original_tools
    ):
        await ctx.output.print(f"Tool '{tool_name}' not found")
        return

    previous_choice = agent._tool_choice
    try:
        agent._tool_choice = tool_name
        result = await ctx.session.send_message(prompt)
        await ctx.output.print(result.content)
    finally:
        agent._tool_choice = previous_choice


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
            ctx.session._agent._pydantic_agent.tool_plain(func)
            await ctx.output.print(f"Tool '{func.__name__}' registered!")

    except Exception as e:  # noqa: BLE001
        await ctx.output.print(f"Error creating tools: {e}")


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

use_tool_cmd = Command(
    name="use-tool",
    description="Execute prompt using a specific tool",
    execute_func=use_tool,
    usage="<tool_name> <prompt>",
    help_text=(
        "Execute a prompt using only the specified tool.\n\n"
        "Example:\n"
        "  /use-tool open_browser 'Open python.org'\n"
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
    name="tool-info",
    description="Show detailed information about a tool",
    execute_func=tool_info,
    usage="<name>",
    help_text=(
        "Display detailed information about a specific tool:\n"
        "- Source (runtime/agent/builtin)\n"
        "- Current status (enabled/disabled)\n"
        "- Description\n"
        "- Schema (for runtime tools)\n\n"
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

register_tool_cmd = Command(
    name="register-tool",
    description="Register a new tool from an import path",
    execute_func=register_tool,
    usage="<import_path> [--name name] [--description desc]",
    help_text=(
        "Register a new tool from a Python import path.\n\n"
        "Arguments:\n"
        "  import_path: Python import path to the function\n"
        "  --name: Optional custom name for the tool\n"
        "  --description: Optional tool description\n\n"
        "Examples:\n"
        "  /register-tool webbrowser.open\n"
        "  /register-tool json.dumps --name format_json\n"
        "  /register-tool os.getcwd --description 'Get current directory'"
    ),
    category="tools",
)
