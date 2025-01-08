from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, TypeVar, overload

from llmling_agent.delegation import interactive_controller
from llmling_agent.delegation.pool import AgentResponse
from llmling_agent.delegation.router import (
    AgentRouter,
    AwaitResponseDecision,
    CallbackRouter,
    Decision,
    EndDecision,
    RouteDecision,
)
from llmling_agent.models.messages import ChatMessage


if TYPE_CHECKING:
    from llmling_agent.agent import Agent, AnyAgent
    from llmling_agent.agent.structured import StructuredAgent
    from llmling_agent.delegation.callbacks import DecisionCallback


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
    ) -> list[AgentResponse[Any]]:
        """Run all agents in parallel."""

        async def run_agent(agent: AnyAgent[TDeps, Any]) -> AgentResponse[Any]:
            try:
                start = time.perf_counter()
                message = await agent.run(
                    prompt or self.shared_prompt, deps=deps or self.shared_deps
                )
                timing = time.perf_counter() - start
                return AgentResponse(
                    agent_name=agent.name, message=message, timing=timing
                )
            except Exception as e:  # noqa: BLE001
                return AgentResponse(
                    agent_name=agent.name,
                    message=ChatMessage(content="", role="assistant"),
                    error=str(e),
                )

        return await asyncio.gather(*[run_agent(a) for a in self.agents])

    async def run_sequential(
        self,
        prompt: str | None = None,
        deps: TDeps | None = None,
    ) -> list[AgentResponse[Any]]:
        """Run agents one after another."""
        results = []
        for agent in self.agents:
            try:
                start = time.perf_counter()
                message = await agent.run(
                    prompt or self.shared_prompt, deps=deps or self.shared_deps
                )
                timing = time.perf_counter() - start
                results.append(
                    AgentResponse(agent_name=agent.name, message=message, timing=timing)
                )
            except Exception as e:  # noqa: BLE001
                results.append(
                    AgentResponse(
                        agent_name=agent.name,
                        message=ChatMessage(content="", role="assistant"),
                        error=str(e),
                    )
                )
        return results

    async def run_controlled(
        self,
        prompt: str | None = None,
        deps: TDeps | None = None,
        *,
        decision_callback: DecisionCallback = interactive_controller,
        router: AgentRouter | None = None,
    ) -> list[AgentResponse[Any]]:
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
            now = time.perf_counter()
            message = await current_agent.run(current_message, deps=actual_deps)
            duration = time.perf_counter() - now
            response = AgentResponse(current_agent.name, message, timing=duration)
            results.append(response)

            # Get next decision
            assert response.message
            decision = await router.decide(response.message.content)

            # Execute the decision
            assert current_agent.context.pool
            await decision.execute(
                response.message, current_agent, current_agent.context.pool
            )

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
                    current_message = str(response.message.content)

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
