"""Agent interaction patterns."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

from pydantic import BaseModel
from typing_extensions import TypeVar

from llmling_agent.delegation.agentgroup import Team
from llmling_agent.delegation.controllers import interactive_controller
from llmling_agent.delegation.pool import AgentPool
from llmling_agent.delegation.router import (
    AgentRouter,
    CallbackRouter,
    ChatMessage,
    Decision,
)
from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from collections.abc import Sequence

    from toprompt import AnyPromptType

    from llmling_agent.agent import Agent, AnyAgent, StructuredAgent
    from llmling_agent.delegation.callbacks import DecisionCallback


logger = get_logger(__name__)
TResult = TypeVar("TResult")
TDeps = TypeVar("TDeps")


class Talk[TDeps, TResult]:
    """Manages agent communication patterns."""

    def __init__(self, agent: AnyAgent[TDeps, TResult]):
        self.agent = agent

    def _resolve_agent(self, target: str | AnyAgent[TDeps, Any]) -> AnyAgent[TDeps, Any]:
        """Resolve string agent name to instance."""
        if isinstance(target, str):
            if not self.agent.context.pool:
                msg = "Pool required for resolving agent names"
                raise ValueError(msg)
            return self.agent.context.pool.get_agent(target)
        return target

    @overload
    async def ask(
        self,
        target: str | Agent[TDeps],
        message: str,
        *,
        include_history: bool = False,
        max_tokens: int | None = None,
    ) -> ChatMessage[str]: ...

    @overload
    async def ask[TOtherResult](
        self,
        target: str | StructuredAgent[TDeps, TOtherResult],
        message: TOtherResult,
        *,
        include_history: bool = False,
        max_tokens: int | None = None,
    ) -> ChatMessage[TOtherResult]: ...

    async def ask[TOtherResult](
        self,
        target: str | AnyAgent[TDeps, TOtherResult],
        message: str | TOtherResult,
        *,
        include_history: bool = False,
        max_tokens: int | None = None,
    ) -> ChatMessage[TOtherResult]:
        """Send message to another agent and wait for response."""
        target_agent = self._resolve_agent(target)

        if include_history:
            history = await self.agent.conversation.format_history(max_tokens=max_tokens)
            await target_agent.conversation.add_context_message(
                history, source=self.agent.name, metadata={"type": "conversation_history"}
            )

        return await target_agent.run(message)

    @overload
    async def controlled(
        self,
        message: str,
        decision_callback: DecisionCallback[str] = interactive_controller,
    ) -> tuple[ChatMessage[str], Decision]: ...

    @overload
    async def controlled(
        self,
        message: TResult,
        decision_callback: DecisionCallback[TResult],
    ) -> tuple[ChatMessage[TResult], Decision]: ...

    async def controlled(
        self,
        message: str | TResult,
        decision_callback: DecisionCallback[Any] = interactive_controller,
        router: AgentRouter | None = None,
    ) -> tuple[ChatMessage[Any], Decision]:
        """Get response with routing decision."""
        assert self.agent.context.pool
        router = router or CallbackRouter(self.agent.context.pool, decision_callback)

        response = await self.agent.run(message)
        decision = await router.decide(response.content)

        return response, decision

    async def pick(
        self,
        from_agents: Team[TDeps] | AgentPool | Sequence[AnyAgent[TDeps, Any]],
        task: str,
        prompt: AnyPromptType | None = None,
    ) -> AnyAgent[TDeps, Any]:
        """Pick most suitable agent for a task.

        Args:
            from_agents: Source to pick from:
                - Team object
                - Agent pool
                - Sequence of agents
            task: Task description
            prompt: Optional custom selection prompt

        Returns:
            Selected agent

        Raises:
            ValueError: If no agents available or selection invalid
        """
        # Get list of agents from various sources
        agents = (
            from_agents.agents
            if isinstance(from_agents, Team)
            else list(from_agents.agents.values())
            if isinstance(from_agents, AgentPool)
            else list(from_agents)
        )

        if not agents:
            msg = "No agents available to pick from"
            raise ValueError(msg)

        descriptions = [agent.__prompt__() for agent in agents]

        default_prompt = f"""Task: {task}

Available agents:
{"-" * 40}
{"\n\n".join(descriptions)}
{"-" * 40}

Based on the task requirements and agent capabilities,
select the most suitable agent to handle this task.

Select ONLY ONE agent by their name."""

        class AgentSelection(BaseModel):
            agent_name: str
            reason: str

        # Use the structured agent wrapper to make the call type-safe
        structured = self.agent.to_structured(AgentSelection)
        result = await structured.run(prompt or default_prompt)

        selected = next((a for a in agents if a.name == result.content.agent_name), None)
        if not selected:
            msg = f"Selected agent {result.content.agent_name} not found in team"
            raise ValueError(msg)
        msg = "Selected %s for task %r. Reason: %s"
        logger.info(msg, selected.name, task, result.content.reason)
        return selected
