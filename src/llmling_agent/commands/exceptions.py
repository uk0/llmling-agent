"""Command-related exceptions."""

from llmling_agent.commands.base import CommandError


class ExitCommandError(CommandError):
    """Special exception to signal clean session exit."""
