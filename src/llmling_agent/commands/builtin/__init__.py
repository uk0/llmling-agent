"""Built-in commands for LLMling agent."""

from llmling_agent.commands.base import BaseCommand
from llmling_agent.commands.builtin.help_cmd import help_cmd, show_command_cmd
from llmling_agent.commands.builtin.meta import meta_cmd
from llmling_agent.commands.builtin.prompts import list_prompts_cmd, prompt_cmd
from llmling_agent.commands.builtin.agents import (
    list_agents_cmd,
    switch_agent_cmd,
    show_agent_cmd,
)
from llmling_agent.commands.builtin.session import clear_cmd, reset_cmd, exit_cmd
from llmling_agent.commands.builtin.env import set_env_cmd, edit_env_cmd
from llmling_agent.commands.builtin.utils import (
    copy_clipboard_cmd,
    edit_agent_file_cmd,
)
from llmling_agent.commands.builtin.tools import (
    list_tools_cmd,
    tool_info_cmd,
    enable_tool_cmd,
    disable_tool_cmd,
    register_tool_cmd,
    write_tool_cmd,
)
from llmling_agent.commands.builtin.models import set_model_cmd
from llmling_agent.commands.builtin.resources import (
    list_resources_cmd,
    show_resource_cmd,
)


def get_builtin_commands() -> list[BaseCommand]:
    """Get list of built-in commands."""
    return [
        help_cmd,
        list_prompts_cmd,
        prompt_cmd,
        switch_agent_cmd,
        list_agents_cmd,
        clear_cmd,
        reset_cmd,
        copy_clipboard_cmd,
        edit_env_cmd,
        list_tools_cmd,
        tool_info_cmd,
        enable_tool_cmd,
        disable_tool_cmd,
        register_tool_cmd,
        set_model_cmd,
        list_resources_cmd,
        show_resource_cmd,
        edit_agent_file_cmd,
        set_env_cmd,
        show_agent_cmd,
        write_tool_cmd,
        meta_cmd,
        show_command_cmd,
        exit_cmd,
    ]
