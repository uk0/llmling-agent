"""Built-in commands for LLMling agent."""

from llmling_agent.commands.base import BaseCommand
from llmling_agent.commands.builtin.hello import hello_command


def get_builtin_commands() -> list[BaseCommand]:
    """Get list of built-in commands."""
    return [
        hello_command,
    ]
