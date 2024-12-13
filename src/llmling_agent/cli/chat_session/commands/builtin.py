from __future__ import annotations

from llmling_agent.cli.chat_session.base import BaseCommand, CommandContext


class HelpCommand(BaseCommand):
    """Show help for available commands."""

    def __init__(self) -> None:
        super().__init__(
            name="help",
            description="Show available commands",
            usage="[command]",
        )

    async def execute(self, ctx: CommandContext) -> None:
        """Show help for all commands or specific command."""
        from llmling_agent.cli.chat_session.commands.builtin import get_builtin_commands

        commands = get_builtin_commands()

        if ctx.args:
            # Show help for specific command
            cmd = next((c for c in commands if c.name == ctx.args), None)
            if not cmd:
                ctx.console.print(f"Unknown command: {ctx.args}", style="red")
                return

            ctx.console.print(f"\nCommand: /{cmd.name}")
            ctx.console.print(f"Description: {cmd.description}")
            if usage := cmd.format_usage():
                ctx.console.print(f"{usage}")
            ctx.console.print()
            return

        # Show all commands
        ctx.console.print("\nAvailable Commands:")
        for cmd in commands:
            ctx.console.print(f"  /{cmd.name:<12} - {cmd.description}")
        ctx.console.print()


class ExitCommand(BaseCommand):
    """Exit the chat session."""

    def __init__(self) -> None:
        super().__init__(
            name="exit",
            description="Exit chat session",
        )

    async def execute(self, ctx: CommandContext) -> None:
        """Exit the session."""
        raise EOFError


def get_builtin_commands() -> list[BaseCommand]:
    """Get list of built-in commands."""
    return [
        HelpCommand(),
        ExitCommand(),
    ]
