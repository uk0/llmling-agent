"""Simple hello command for testing."""

from __future__ import annotations

from llmling_agent.commands.base import Command, CommandContext


async def help_command(
    ctx: CommandContext,
    args: list[str],
    kwargs: dict[str, str],
) -> None:
    """Show available commands."""
    store = ctx.session._command_store

    if args:  # Detail for specific command
        name = args[0]
        if cmd := store.get_command(name):
            await ctx.output.print(
                f"Command: /{cmd.name}\n"
                f"Description: {cmd.description}\n"
                f"{cmd.format_usage() or ''}"
            )
        else:
            await ctx.output.print(f"Unknown command: {name}")
        return

    # Simple flat list of commands
    await ctx.output.print("\nAvailable commands:")
    for cmd in store.list_commands():
        await ctx.output.print(f"  /{cmd.name:<12} - {cmd.description}")


help_cmd = Command(
    name="help",
    description="Show available commands",
    execute_func=help_command,
    usage="[command]",
    help_text=(
        "Display help information about commands.\n\n"
        "Usage:\n"
        "  /help         - List all available commands\n"
        "  /help <cmd>   - Show detailed help for a command\n\n"
        "Example: /help register-tool"
    ),
    category="system",
)
