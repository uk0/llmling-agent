"""Agent capabilities definition."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from llmling import LLMCallableTool, ToolError
from psygnal import EventedModel
from pydantic import ConfigDict
from pydantic_ai import RunContext  # noqa: TC002

from llmling_agent.models.context import AgentContext  # noqa: TC001


if TYPE_CHECKING:
    from llmling_agent.agent.agent import LLMlingAgent


AccessLevel = Literal["none", "own", "all"]


def create_delegate_tool() -> LLMCallableTool:
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

    return LLMCallableTool.from_callable(delegate_to)


def create_list_agents_tool() -> LLMCallableTool:
    async def list_available_agents(ctx: RunContext[AgentContext]) -> list[str]:
        if not ctx.deps.pool:
            msg = "Agent needs to be in a pool to list agents"
            raise ToolError(msg)
        return ctx.deps.pool.list_agents()

    return LLMCallableTool.from_callable(list_available_agents)


class Capabilities(EventedModel):
    """Defines what operations an agent is allowed to perform.

    Controls an agent's permissions and access levels including:
    - Agent discovery and delegation abilities
    - History access permissions
    - Statistics viewing rights
    - Tool usage restrictions

    Can be defined per role or customized per agent.
    """

    can_list_agents: bool = False
    """Whether the agent can discover other available agents."""

    can_delegate_tasks: bool = False
    """Whether the agent can delegate tasks to other agents."""

    can_observe_agents: bool = False
    """Whether the agent can monitor other agents' activities."""

    history_access: AccessLevel = "none"
    """Level of access to conversation history.

    Levels:
    - none: No access to history
    - own: Can only access own conversations
    - all: Can access all agents' conversations
    """

    stats_access: AccessLevel = "none"
    """Level of access to usage statistics.

    Levels:
    - none: No access to statistics
    - own: Can only view own statistics
    - all: Can view all agents' statistics
    """

    def has_capability(self, capability: str) -> bool:
        """Check if a specific capability is enabled.

        Args:
            capability: Name of capability to check.
                      Can be a boolean capability (e.g., "can_delegate_tasks")
                      or an access level (e.g., "history_access")

        Returns:
            True if capability is enabled
        """
        match capability:
            case str() if hasattr(self, capability):
                value = getattr(self, capability)
                return bool(value) if isinstance(value, bool) else value != "none"
            case _:
                msg = f"Unknown capability: {capability}"
                raise ValueError(msg)

    def register_delegation_tools(self, agent: LLMlingAgent[Any, Any]):
        """Register delegation tools if enabled."""
        if self.can_delegate_tasks:
            tool = create_delegate_tool()
            agent.tools.register_tool(tool, enabled=True, source="builtin")
        if self.can_list_agents:
            tool = create_list_agents_tool()
            agent.tools.register_tool(tool, enabled=True, source="builtin")

    def enable(self, capability: str):
        """Enable a capability."""
        setattr(self, capability, True)

    def disable(self, capability: str):
        """Disable a capability."""
        setattr(self, capability, False)

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)
