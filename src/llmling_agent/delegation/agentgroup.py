from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, TypeVar, overload

from llmling_agent.delegation import interactive_controller
from llmling_agent.delegation.router import (
    AgentRouter,
    AwaitResponseDecision,
    CallbackRouter,
    Decision,
    EndDecision,
    RouteDecision,
)


if TYPE_CHECKING:
    from llmling_agent.agent import Agent, AnyAgent
    from llmling_agent.agent.structured import StructuredAgent
    from llmling_agent.delegation.callbacks import DecisionCallback
    from llmling_agent.models.messages import ChatMessage


TDeps = TypeVar("TDeps")


class AgentGroup[TDeps]:
    """Group of agents that can execute together."""

    def __init__(
        self,
        agents: list[AnyAgent[TDeps, Any]],
        *,
        shared_prompt: str | None = None,
        shared_deps: TDeps | None = None,
    ):
        self.agents = agents
        self.shared_prompt = shared_prompt
        self.shared_deps = shared_deps

    async def run_parallel(
        self,
        prompt: str | None = None,
        deps: TDeps | None = None,
    ) -> list[ChatMessage[Any]]:
        """Run all agents in parallel."""
        actual_prompt = prompt or self.shared_prompt
        actual_deps = deps or self.shared_deps
        tasks = [agent.run(actual_prompt, deps=actual_deps) for agent in self.agents]
        return await asyncio.gather(*tasks)

    async def run_sequential(
        self,
        prompt: str | None = None,
        deps: TDeps | None = None,
    ) -> list[ChatMessage[Any]]:
        """Run agents one after another."""
        actual_prompt = prompt or self.shared_prompt
        actual_deps = deps or self.shared_deps
        return [await agent.run(actual_prompt, deps=actual_deps) for agent in self.agents]

    async def run_controlled(
        self,
        prompt: str | None = None,
        deps: TDeps | None = None,
        *,
        decision_callback: DecisionCallback = interactive_controller,
        router: AgentRouter | None = None,
    ) -> list[ChatMessage[Any]]:
        """Run with explicit control over agent interactions."""
        results = []
        actual_prompt = prompt or self.shared_prompt
        actual_deps = deps or self.shared_deps

        # Create router for decisions
        assert self.agents[0].context.pool
        router = router or CallbackRouter(self.agents[0].context.pool, decision_callback)
        current_agent = self.agents[0]
        current_message = actual_prompt

        while True:
            # Get response from current agent
            response = await current_agent.run(current_message, deps=actual_deps)
            results.append(response)

            # Get next decision
            decision = await router.decide(response.content)

            # Execute the decision
            assert current_agent.context.pool
            await decision.execute(response, current_agent, current_agent.context.pool)

            match decision:
                case EndDecision():
                    break
                case RouteDecision():
                    continue
                case AwaitResponseDecision():
                    current_agent = next(
                        (a for a in self.agents if a.name == decision.target_agent),
                        current_agent,
                    )
                    current_message = str(response.content)

        return results

    @overload
    async def controlled_talk[TResult](
        self,
        agent: StructuredAgent[TDeps, TResult],
        message: TResult,
        *,
        decision_callback: DecisionCallback[TResult],
    ) -> tuple[ChatMessage[TResult], Decision]: ...

    @overload
    async def controlled_talk(
        self,
        agent: Agent[TDeps],
        message: str,
        *,
        decision_callback: DecisionCallback[str] = interactive_controller,
    ) -> tuple[ChatMessage[str], Decision]: ...

    async def controlled_talk(
        self,
        agent: AnyAgent[TDeps, Any],
        message: Any,
        *,
        decision_callback: DecisionCallback[Any] = interactive_controller,
        router: AgentRouter | None = None,
    ) -> tuple[ChatMessage[Any], Decision]:
        """Get one response with control decision.

        Args:
            agent: Agent to use
            message: Message to send
            decision_callback: Callback for routing decision
            router: Optional router to use

        Returns:
            Tuple of (response message, routing decision)
        """
        # Use existing router system
        assert agent.context.pool
        router = router or CallbackRouter(agent.context.pool, decision_callback)
        response = await agent.run(message)
        decision = await router.decide(response.content)
        return response, decision
