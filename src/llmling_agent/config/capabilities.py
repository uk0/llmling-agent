"""Agent capabilities definition."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from llmling import LLMCallableTool
from psygnal import EventedModel
from pydantic import ConfigDict


if TYPE_CHECKING:
    from llmling_agent.agent.agent import Agent


AccessLevel = Literal["none", "own", "all"]


class Capabilities(EventedModel):
    """Defines what operations an agent is allowed to perform."""

    # Agent discovery and delegation
    can_list_agents: bool = False
    """Whether the agent can discover other available agents."""

    can_delegate_tasks: bool = False
    """Whether the agent can delegate tasks to other agents."""

    can_observe_agents: bool = False
    """Whether the agent can monitor other agents' activities."""

    # History and statistics access
    history_access: AccessLevel = "none"
    """Level of access to conversation history."""

    stats_access: AccessLevel = "none"
    """Level of access to usage statistics."""

    # Resource capabilities
    can_load_resources: bool = False
    """Whether the agent can load and access resource content."""

    can_list_resources: bool = False
    """Whether the agent can discover available resources."""

    # Tool management
    can_register_tools: bool = False
    """Whether the agent can register importable functions as tools."""

    can_register_code: bool = False
    """Whether the agent can create new tools from provided code."""

    can_install_packages: bool = False
    """Whether the agent can install Python packages for tools."""

    can_chain_tools: bool = False
    """Whether the agent can chain multiple tool calls into one."""

    # Execution

    can_execute_code: bool = False
    """Whether the agent can execute Python code (WARNING: No sandbox)."""

    can_execute_commands: bool = False
    """Whether the agent can execute CLI commands (use at your own risk)."""

    # Agent creation
    can_create_workers: bool = False
    """Whether the agent can create worker agents (as tools)."""

    can_create_delegates: bool = False
    """Whether the agent can spawn temporary delegate agents."""

    can_add_agents: bool = False
    """Whether the agent can add aother agents to the pool."""

    can_ask_agents: bool = False
    """Whether the agent can ask other agents of the pool."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)

    def __contains__(self, required: Capabilities) -> bool:
        """Check if these capabilities contain all required capabilities.

        Example:
            required in agent.capabilities  # Can agent fulfill requirements?
        """
        # Check all boolean capabilities
        for field in self.__fields__:
            if isinstance(getattr(required, field), bool):  # noqa: SIM102
                if getattr(required, field) and not getattr(self, field):
                    return False

        # Check access levels (none < own < all)
        access_order = {"none": 0, "own": 1, "all": 2}
        for field in ("history_access", "stats_access"):
            required_level = access_order[getattr(required, field)]
            self_level = access_order[getattr(self, field)]
            if required_level > self_level:
                return False

        return True

    def has_capability(self, capability: str) -> bool:
        """Check if a specific capability is enabled.

        Args:
            capability: Name of capability to check.
                      Can be a boolean capability (e.g., "can_delegate_tasks")
                      or an access level (e.g., "history_access")
        """
        match capability:
            case str() if hasattr(self, capability):
                value = getattr(self, capability)
                return bool(value) if isinstance(value, bool) else value != "none"
            case _:
                msg = f"Unknown capability: {capability}"
                raise ValueError(msg)

    def register_capability_tools(self, agent: Agent[Any]):
        """Register all capability-based tools."""
        # Resource tools - always register, enable based on capabilities
        agent.tools.register_tool(
            LLMCallableTool.from_callable(agent.runtime.load_resource),
            enabled=self.can_load_resources,
            source="builtin",
            requires_capability="can_load_resources",
        )
        agent.tools.register_tool(
            LLMCallableTool.from_callable(agent.runtime.get_resources),
            enabled=self.can_list_resources,
            source="builtin",
            requires_capability="can_list_resources",
        )

        # Tool management
        agent.tools.register_tool(
            LLMCallableTool.from_callable(agent.runtime.register_tool),
            enabled=self.can_register_tools,
            source="builtin",
            requires_capability="can_register_tools",
        )
        agent.tools.register_tool(
            LLMCallableTool.from_callable(agent.runtime.register_code_tool),
            enabled=self.can_register_code,
            source="builtin",
            requires_capability="can_register_code",
        )
        agent.tools.register_tool(
            LLMCallableTool.from_callable(agent.runtime.install_package),
            enabled=self.can_install_packages,
            source="builtin",
            requires_capability="can_install_packages",
        )
        from llmling_agent.tools import capability_tools

        # Agent discovery and delegation
        agent.tools.register_tool(
            LLMCallableTool.from_callable(capability_tools.create_worker_agent),
            enabled=self.can_create_workers,
            source="builtin",
            requires_capability="can_create_workers",
        )
        agent.tools.register_tool(
            LLMCallableTool.from_callable(capability_tools.spawn_delegate),
            enabled=self.can_create_delegates,
            source="builtin",
            requires_capability="can_create_delegates",
        )
        agent.tools.register_tool(
            LLMCallableTool.from_callable(capability_tools.list_available_agents),
            enabled=self.can_list_agents,
            source="builtin",
            requires_capability="can_list_agents",
        )
        agent.tools.register_tool(
            LLMCallableTool.from_callable(capability_tools.delegate_to),
            enabled=self.can_delegate_tasks,
            source="builtin",
            requires_capability="can_delegate_tasks",
        )
        # History tools
        agent.tools.register_tool(
            LLMCallableTool.from_callable(capability_tools.search_history),
            enabled=self.history_access != "none",
            source="builtin",
            requires_capability="history_access",
        )
        agent.tools.register_tool(
            LLMCallableTool.from_callable(capability_tools.show_statistics),
            enabled=self.stats_access != "none",
            source="builtin",
            requires_capability="stats_access",
        )
        agent.tools.register_tool(
            LLMCallableTool.from_callable(capability_tools.add_agent),
            enabled=self.can_add_agents,
            source="builtin",
            requires_capability="can_add_agents",
        )
        agent.tools.register_tool(
            LLMCallableTool.from_callable(capability_tools.ask_agent),
            enabled=self.can_ask_agents,
            source="builtin",
            requires_capability="can_ask_agents",
        )
