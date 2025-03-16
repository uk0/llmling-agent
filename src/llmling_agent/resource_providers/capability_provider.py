"""Resource provider for agent capabilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llmling_agent.resource_providers.base import ResourceProvider
from llmling_agent.tools.base import Tool


if TYPE_CHECKING:
    from llmling import RuntimeConfig

    from llmling_agent.config.capabilities import Capabilities


class CapabilitiesResourceProvider(ResourceProvider):
    """Provides tools based on agent capabilities."""

    requires_async: bool = False

    def __init__(
        self,
        capabilities: Capabilities,
        runtime: RuntimeConfig | None = None,
        name: str = "capability_tools",
    ):
        super().__init__(name)
        self.capabilities = capabilities
        self.runtime = runtime

    async def get_tools(self) -> list[Tool]:  # noqa: PLR0915
        """Get all tools enabled by current capabilities."""
        tools: list[Tool] = []

        # Resource tools (require runtime)
        if self.runtime:
            if self.capabilities.can_load_resources:
                tool = Tool.from_callable(
                    self.runtime.load_resource,
                    source="builtin",
                    requires_capability="can_load_resources",
                )
                tools.append(tool)
            if self.capabilities.can_list_resources:
                tool = Tool.from_callable(
                    self.runtime.get_resources,
                    source="builtin",
                    requires_capability="can_list_resources",
                )
                tools.append(tool)
            if self.capabilities.can_register_tools:
                tool = Tool.from_callable(
                    self.runtime.register_tool,
                    source="builtin",
                    requires_capability="can_register_tools",
                )
                tools.append(tool)
            if self.capabilities.can_register_code:
                tool = Tool.from_callable(
                    self.runtime.register_code_tool,
                    source="builtin",
                    requires_capability="can_register_code",
                )
                tools.append(tool)
            if self.capabilities.can_install_packages:
                tool = Tool.from_callable(
                    self.runtime.install_package,
                    source="builtin",
                    requires_capability="can_install_packages",
                )
                tools.append(tool)

        # Agent/team tools
        from llmling_agent_tools import capability_tools

        if self.capabilities.can_create_workers:
            tool = Tool.from_callable(
                capability_tools.create_worker_agent,
                source="builtin",
                requires_capability="can_create_workers",
            )
            tools.append(tool)
        if self.capabilities.can_create_delegates:
            tool = Tool.from_callable(
                capability_tools.spawn_delegate,
                source="builtin",
                requires_capability="can_create_delegates",
            )
            tools.append(tool)
        if self.capabilities.can_list_agents:
            tool = Tool.from_callable(
                capability_tools.list_available_agents,
                source="builtin",
                requires_capability="can_list_agents",
            )
            tools.append(tool)
        if self.capabilities.can_list_teams:
            tool = Tool.from_callable(
                capability_tools.list_available_teams,
                source="builtin",
                requires_capability="can_list_teams",
            )
            tools.append(tool)
        if self.capabilities.can_delegate_tasks:
            tool = Tool.from_callable(
                capability_tools.delegate_to,
                source="builtin",
                requires_capability="can_delegate_tasks",
            )
            tools.append(tool)

        # History and stats tools
        if self.capabilities.history_access != "none":
            tool = Tool.from_callable(
                capability_tools.search_history,
                source="builtin",
                requires_capability="history_access",
            )
            tools.append(tool)
        if self.capabilities.stats_access != "none":
            tool = Tool.from_callable(
                capability_tools.show_statistics,
                source="builtin",
                requires_capability="stats_access",
            )
            tools.append(tool)

        # Agent/team management
        if self.capabilities.can_add_agents:
            tool = Tool.from_callable(
                capability_tools.add_agent,
                source="builtin",
                requires_capability="can_add_agents",
            )
            tools.append(tool)
        if self.capabilities.can_add_teams:
            tool = Tool.from_callable(
                capability_tools.add_team,
                source="builtin",
                requires_capability="can_add_teams",
            )
            tools.append(tool)

        if self.capabilities.can_connect_nodes:
            tool = Tool.from_callable(
                capability_tools.connect_nodes,
                source="builtin",
                requires_capability="can_can_connect_nodes",
            )
            tools.append(tool)

        if self.capabilities.can_ask_agents:
            tool = Tool.from_callable(
                capability_tools.ask_agent,
                source="builtin",
                requires_capability="can_ask_agents",
            )
            tools.append(tool)
        if self.capabilities.can_read_files:
            tool = Tool.from_callable(
                capability_tools.read_file,
                source="builtin",
                requires_capability="can_read_files",
            )
            tools.append(tool)
        if self.capabilities.can_list_directories:
            tool = Tool.from_callable(
                capability_tools.list_directory,
                source="builtin",
                requires_capability="can_list_directories",
            )
            tools.append(tool)

        return tools
