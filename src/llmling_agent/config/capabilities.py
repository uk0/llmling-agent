"""Agent capabilities definition."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from llmling import LLMCallableTool, ToolError
from psygnal import EventedModel
from pydantic import ConfigDict
from pydantic_ai import RunContext  # noqa: TC002

from llmling_agent.models.context import AgentContext  # noqa: TC001


if TYPE_CHECKING:
    from llmling_agent.agent.agent import LLMlingAgent


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

    # Agent creation
    can_create_workers: bool = False
    """Whether the agent can create worker agents (as tools)."""

    can_create_delegates: bool = False
    """Whether the agent can spawn temporary delegate agents."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)

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

    def register_capability_tools(self, agent: LLMlingAgent[Any, Any]):
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

        # Agent discovery and delegation
        agent.tools.register_tool(
            LLMCallableTool.from_callable(self.create_worker_agent),
            enabled=self.can_create_workers,
            source="builtin",
            requires_capability="can_create_workers",
        )
        agent.tools.register_tool(
            LLMCallableTool.from_callable(self.spawn_delegate),
            enabled=self.can_create_delegates,
            source="builtin",
            requires_capability="can_create_delegates",
        )
        agent.tools.register_tool(
            LLMCallableTool.from_callable(self.list_available_agents),
            enabled=self.can_list_agents,
            source="builtin",
            requires_capability="can_list_agents",
        )
        agent.tools.register_tool(
            LLMCallableTool.from_callable(self.delegate_to),
            enabled=self.can_delegate_tasks,
            source="builtin",
            requires_capability="can_delegate_tasks",
        )

    # IMPLEMENTATIONS
    @staticmethod
    async def delegate_to(
        ctx: RunContext[AgentContext],
        agent_name: str,
        prompt: str,
    ) -> str:
        if not ctx.deps.pool:
            msg = "Agent needs to be in a pool to delegate tasks"
            raise ToolError(msg)
        specialist = ctx.deps.pool.get_agent(agent_name)
        result = await specialist.run(prompt)
        return str(result.data)

    @staticmethod
    async def list_available_agents(ctx: RunContext[AgentContext]) -> list[str]:
        if not ctx.deps.pool:
            msg = "Agent needs to be in a pool to list agents"
            raise ToolError(msg)
        return ctx.deps.pool.list_agents()

    @staticmethod
    async def create_worker_agent(
        ctx: RunContext[AgentContext],
        name: str,
        system_prompt: str,
        model: str | None = None,
    ) -> str:
        """Create a new agent and register it as a tool.

        The new agent will be available as a tool for delegating specific tasks.
        It inherits the current model unless overridden.
        """
        from llmling_agent.models.agents import AgentConfig

        if not ctx.deps.pool:
            msg = "No agent pool available"
            raise ToolError(msg)

        config = AgentConfig(
            name=name, system_prompts=[system_prompt], model=model or ctx.model.name()
        )

        worker = await ctx.deps.pool.create_agent(name, config)
        assert ctx.deps.agent
        tool_info = ctx.deps.agent.register_worker(worker)
        return f"Created worker agent and registered as tool: {tool_info.name}"

    @staticmethod
    async def spawn_delegate(
        ctx: RunContext[AgentContext],
        task: str,
        system_prompt: str,
        model: str | None = None,
        capabilities: dict[str, bool] | None = None,
        connect_back: bool = False,
    ) -> str:
        """Spawn a temporary agent for a specific task.

        Creates an ephemeral agent that will execute the task and clean up automatically.
        Optionally connects back to receive results.
        """
        from llmling_agent.models.agents import AgentConfig

        if not ctx.deps.pool:
            msg = "No agent pool available"
            raise ToolError(msg)

        name = f"delegate_{uuid4().hex[:8]}"
        config = AgentConfig(
            name=name, system_prompts=[system_prompt], model=model or ctx.model.name()
        )

        agent = await ctx.deps.pool.create_agent(name, config, temporary=True)
        if connect_back:
            assert ctx.deps.agent
            ctx.deps.agent.pass_results_to(agent)
        await agent.run(task)
        return f"Spawned delegate {name} for task"
