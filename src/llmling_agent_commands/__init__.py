"""Built-in commands for LLMling agent."""

from slashed import BaseCommand, SlashedCommand

from llmling_agent_commands.agents import (
    create_agent_cmd,
    list_agents_cmd,
    show_agent_cmd,
    switch_agent_cmd,
)
from llmling_agent_commands.connections import (
    connect_cmd,
    disconnect_cmd,
    connections_cmd,
    disconnect_all_cmd,
)
from llmling_agent_commands.env import edit_env_cmd, set_env_cmd
from llmling_agent_commands.meta import meta_cmd
from llmling_agent_commands.models import set_model_cmd
from llmling_agent_commands.prompts import list_prompts_cmd, prompt_cmd
from llmling_agent_commands.resources import (
    list_resources_cmd,
    show_resource_cmd,
    add_resource_cmd,
)
from llmling_agent_commands.session import clear_cmd, reset_cmd
from llmling_agent_commands.read import read_cmd
from llmling_agent_commands.tools import (
    disable_tool_cmd,
    enable_tool_cmd,
    list_tools_cmd,
    register_tool_cmd,
    tool_info_cmd,
    # write_tool_cmd,
)
from llmling_agent_commands.workers import (
    add_worker_cmd,
    remove_worker_cmd,
    list_workers_cmd,
)
from llmling_agent_commands.utils import CopyClipboardCommand, EditAgentFileCommand


def get_commands() -> list[BaseCommand | type[SlashedCommand]]:
    """Get list of built-in commands."""
    return [
        list_prompts_cmd,
        prompt_cmd,
        create_agent_cmd,
        switch_agent_cmd,
        list_agents_cmd,
        connect_cmd,
        disconnect_cmd,
        connections_cmd,
        disconnect_all_cmd,
        clear_cmd,
        reset_cmd,
        CopyClipboardCommand,
        edit_env_cmd,
        list_tools_cmd,
        tool_info_cmd,
        enable_tool_cmd,
        disable_tool_cmd,
        register_tool_cmd,
        set_model_cmd,
        add_resource_cmd,
        list_resources_cmd,
        show_resource_cmd,
        EditAgentFileCommand,
        set_env_cmd,
        show_agent_cmd,
        # write_tool_cmd,
        add_worker_cmd,
        remove_worker_cmd,
        list_workers_cmd,
        meta_cmd,
        read_cmd,
    ]
