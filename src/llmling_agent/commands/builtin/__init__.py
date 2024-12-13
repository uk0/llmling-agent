"""Built-in commands for LLMling agent."""

from llmling_agent.commands.base import BaseCommand
from llmling_agent.commands.builtin.hello import hello_command
from llmling_agent.commands.builtin.help_cmd import help_cmd
from llmling_agent.commands.builtin.prompts import list_prompts_cmd, prompt_cmd
from llmling_agent.commands.builtin.agents import list_agents_cmd, switch_agent_cmd


def get_builtin_commands() -> list[BaseCommand]:
    """Get list of built-in commands."""
    return [
        hello_command,
        help_cmd,
        list_prompts_cmd,
        prompt_cmd,
        switch_agent_cmd,
        list_agents_cmd,
    ]
