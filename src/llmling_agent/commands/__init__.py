"""Built-in commands for LLMling agent."""

from slashed import BaseCommand

from llmling_agent.commands.agents import (
    list_agents_cmd,
    show_agent_cmd,
    switch_agent_cmd,
)
from llmling_agent.commands.env import edit_env_cmd, set_env_cmd
from llmling_agent.commands.meta import meta_cmd
from llmling_agent.commands.models import set_model_cmd
from llmling_agent.commands.prompts import list_prompts_cmd, prompt_cmd
from llmling_agent.commands.resources import (
    list_resources_cmd,
    show_resource_cmd,
)
from llmling_agent.commands.session import clear_cmd, reset_cmd
from llmling_agent.commands.tools import (
    disable_tool_cmd,
    enable_tool_cmd,
    list_tools_cmd,
    register_tool_cmd,
    tool_info_cmd,
    write_tool_cmd,
)
from llmling_agent.commands.utils import (
    copy_clipboard_cmd,
    edit_agent_file_cmd,
)


def get_commands() -> list[BaseCommand]:
    """Get list of built-in commands."""
    return [
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
    ]
