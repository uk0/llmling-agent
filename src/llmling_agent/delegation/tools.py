"""Tools for agent delegation and collaboration."""

from __future__ import annotations  # noqa: I001

from typing import TYPE_CHECKING, Any

from llmling_agent.log import get_logger
from pydantic_ai import RunContext  # noqa: TC002
from llmling_agent.models import AgentContext  # noqa: TC001


if TYPE_CHECKING:
    from collections.abc import Callable

    from llmling_agent import LLMlingAgent
    from llmling_agent.delegation.pool import AgentPool


logger = get_logger(__name__)


class DelegationTools:
    """Tools for agent delegation and collaboration."""

    def __init__(self, pool: AgentPool):
        """Initialize delegation tools.

        Args:
            pool: Agent pool for collaboration
        """
        self.pool = pool

    async def delegate_to(
        self,
        ctx: RunContext[AgentContext],
        agent_name: str,
        prompt: str,
    ) -> str:
        """Delegate a task to another agent.

        Args:
            ctx: Run context with agent capabilities
            agent_name: Name of agent to delegate to
            prompt: Task to delegate

        Returns:
            Agent's response

        Raises:
            PermissionError: If agent cannot delegate tasks
            ValueError: If target agent not found
        """
        if not ctx.deps.capabilities.can_delegate_tasks:
            msg = "This agent cannot delegate tasks"
            raise PermissionError(msg)

        logger.info("Delegating to %s: %s", agent_name, prompt)
        specialist: LLMlingAgent[Any, str] = self.pool.get_agent(agent_name)
        result = await specialist.run(prompt)
        return str(result.data)

    async def list_available_agents(
        self,
        ctx: RunContext[AgentContext],
    ) -> list[str]:
        """List available agents for delegation.

        Args:
            ctx: Run context with agent capabilities

        Returns:
            List of available agent names

        Raises:
            PermissionError: If agent cannot list other agents
        """
        if not ctx.deps.capabilities.can_list_agents:
            msg = "This agent cannot list other agents"
            raise PermissionError(msg)

        return self.pool.list_agents()

    async def brainstorm(
        self,
        ctx: RunContext[AgentContext],
        prompt: str,
        team: list[str] | None = None,
        *,
        rounds: int = 3,
    ) -> list[str]:
        """Run a collaborative brainstorming session.

        Multiple agents work together to generate and build upon ideas.

        Args:
            ctx: Run context with agent capabilities
            prompt: Topic to brainstorm about
            team: List of agent names to participate, if None uses all available
            rounds: Number of brainstorming rounds

        Returns:
            List of generated ideas

        Raises:
            PermissionError: If agent cannot delegate tasks
        """
        if not ctx.deps.capabilities.can_delegate_tasks:
            msg = "This agent cannot delegate tasks"
            raise PermissionError(msg)

        # Get team or all available agents
        available_agents = await self.list_available_agents(ctx)
        actual_team = team or available_agents

        ideas: list[str] = []
        for round_num in range(rounds):
            round_prompt = (
                f"Round {round_num + 1}/{rounds}. Previous ideas:\n"
                f"{'. '.join(ideas)}\n\n"
                f"Original task: {prompt}\n"
                "Please build on these ideas or add new ones."
            )

            responses = await self.pool.team_task(round_prompt, actual_team)
            ideas = [resp.response for resp in responses if resp.success]
        return ideas

    async def debate(
        self,
        ctx: RunContext[AgentContext],
        topic: str,
        team: list[str],
        *,
        rounds: int = 3,
    ) -> list[str]:
        """Conduct a structured debate between agents.

        Agents take turns presenting arguments and counter-arguments.

        Args:
            ctx: Run context with agent capabilities
            topic: Topic to debate
            team: List of participating agents
            rounds: Number of debate rounds

        Returns:
            List of debate contributions

        Raises:
            PermissionError: If agent cannot delegate tasks
        """
        if not ctx.deps.capabilities.can_delegate_tasks:
            msg = "This agent cannot delegate tasks"
            raise PermissionError(msg)

        discussion: list[str] = []
        for round_num in range(rounds):
            round_prompt = (
                f"Round {round_num + 1}/{rounds} of debate on: {topic}\n"
                f"Previous discussion:\n{'. '.join(discussion)}\n\n"
                "Please provide your perspective, addressing previous points."
            )

            for agent_name in team:
                specialist: LLMlingAgent[Any, str] = self.pool.get_agent(agent_name)
                result = await specialist.run(round_prompt)
                discussion.append(f"{agent_name}: {result.data}")

        return discussion

    async def critique(
        self,
        ctx: RunContext[AgentContext],
        content: str,
        critic_name: str,
        *,
        max_rounds: int = 3,
    ) -> list[str]:
        """Have a critic agent review and improve content iteratively.

        Args:
            ctx: Run context with agent capabilities
            content: Content to review and improve
            critic_name: Name of the critic agent
            max_rounds: Maximum number of improvement rounds

        Returns:
            List of improvements and final version

        Raises:
            PermissionError: If agent cannot delegate tasks
        """
        if not ctx.deps.capabilities.can_delegate_tasks:
            msg = "This agent cannot delegate tasks"
            raise PermissionError(msg)

        critic: LLMlingAgent[Any, str] = self.pool.get_agent(critic_name)
        current_version = content
        improvements: list[str] = []

        for round_num in range(max_rounds):
            prompt = (
                f"Review round {round_num + 1}/{max_rounds}\n"
                "Current version:\n"
                f"{current_version}\n\n"
                "Please provide specific improvements or indicate if no further "
                "improvements needed."
            )

            result = await critic.run(prompt)
            improvement = result.data

            if "no further improvements" in improvement.lower():
                break

            improvements.append(improvement)
            current_version = improvement

        return improvements

    @property
    def tools(self) -> list[Callable]:
        """Get all tools."""
        return [
            self.delegate_to,
            self.list_available_agents,
            self.brainstorm,
            self.debate,
        ]


def register_delegation_tools(agent: LLMlingAgent[Any, Any], pool: AgentPool):
    """Register all delegation tools with an agent.

    Args:
        agent: Agent to register tools with
        pool: Agent pool for collaboration
    """
    tools = DelegationTools(pool)

    delegation_functions: list[Callable[..., Any]] = [
        tools.delegate_to,
        tools.list_available_agents,
        tools.brainstorm,
        tools.debate,
    ]

    for func in delegation_functions:
        agent.tools.register_tool(
            func, enabled=True, source="dynamic", metadata={"type": "delegation"}
        )
