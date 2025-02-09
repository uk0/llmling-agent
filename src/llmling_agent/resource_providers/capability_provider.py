from __future__ import annotations

from typing import TYPE_CHECKING

from llmling_agent.resource_providers.base import ResourceProvider
from llmling_agent.tools.base import ToolInfo


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
    ) -> None:
        super().__init__(name)
        self.capabilities = capabilities
        self.runtime = runtime

    async def get_tools(self) -> list[ToolInfo]:
        """Get all tools enabled by current capabilities."""
        tools: list[ToolInfo] = []

        # Resource tools (require runtime)
        if self.runtime:
            if self.capabilities.can_load_resources:
                tools.append(
                    ToolInfo.from_callable(
                        self.runtime.load_resource,
                        source="builtin",
                        requires_capability="can_load_resources",
                    )
                )
            if self.capabilities.can_list_resources:
                tools.append(
                    ToolInfo.from_callable(
                        self.runtime.get_resources,
                        source="builtin",
                        requires_capability="can_list_resources",
                    )
                )
            if self.capabilities.can_register_tools:
                tools.append(
                    ToolInfo.from_callable(
                        self.runtime.register_tool,
                        source="builtin",
                        requires_capability="can_register_tools",
                    )
                )
            if self.capabilities.can_register_code:
                tools.append(
                    ToolInfo.from_callable(
                        self.runtime.register_code_tool,
                        source="builtin",
                        requires_capability="can_register_code",
                    )
                )
            if self.capabilities.can_install_packages:
                tools.append(
                    ToolInfo.from_callable(
                        self.runtime.install_package,
                        source="builtin",
                        requires_capability="can_install_packages",
                    )
                )

        # Agent/team tools
        from llmling_agent_tools import capability_tools

        if self.capabilities.can_create_workers:
            tools.append(
                ToolInfo.from_callable(
                    capability_tools.create_worker_agent,
                    source="builtin",
                    requires_capability="can_create_workers",
                )
            )
        if self.capabilities.can_create_delegates:
            tools.append(
                ToolInfo.from_callable(
                    capability_tools.spawn_delegate,
                    source="builtin",
                    requires_capability="can_create_delegates",
                )
            )
        if self.capabilities.can_list_agents:
            tools.append(
                ToolInfo.from_callable(
                    capability_tools.list_available_agents,
                    source="builtin",
                    requires_capability="can_list_agents",
                )
            )
        if self.capabilities.can_list_teams:
            tools.append(
                ToolInfo.from_callable(
                    capability_tools.list_available_teams,
                    source="builtin",
                    requires_capability="can_list_teams",
                )
            )
        if self.capabilities.can_delegate_tasks:
            tools.append(
                ToolInfo.from_callable(
                    capability_tools.delegate_to,
                    source="builtin",
                    requires_capability="can_delegate_tasks",
                )
            )

        # History and stats tools
        if self.capabilities.history_access != "none":
            tools.append(
                ToolInfo.from_callable(
                    capability_tools.search_history,
                    source="builtin",
                    requires_capability="history_access",
                )
            )
        if self.capabilities.stats_access != "none":
            tools.append(
                ToolInfo.from_callable(
                    capability_tools.show_statistics,
                    source="builtin",
                    requires_capability="stats_access",
                )
            )

        # Agent/team management
        if self.capabilities.can_add_agents:
            tools.append(
                ToolInfo.from_callable(
                    capability_tools.add_agent,
                    source="builtin",
                    requires_capability="can_add_agents",
                )
            )
        if self.capabilities.can_add_teams:
            tools.append(
                ToolInfo.from_callable(
                    capability_tools.add_team,
                    source="builtin",
                    requires_capability="can_add_teams",
                )
            )

        if self.capabilities.can_connect_nodes:
            tools.append(
                ToolInfo.from_callable(
                    capability_tools.connect_nodes,
                    source="builtin",
                    requires_capability="can_can_connect_nodes",
                )
            )

        if self.capabilities.can_ask_agents:
            tools.append(
                ToolInfo.from_callable(
                    capability_tools.ask_agent,
                    source="builtin",
                    requires_capability="can_ask_agents",
                )
            )

        return tools
