from __future__ import annotations

import asyncio
from datetime import datetime
import time
from typing import TYPE_CHECKING, Any, TypeVar, overload

from psygnal.containers import EventedList

from llmling_agent.agent.connection import TalkManager, TeamTalk
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


class TeamResponse(list[AgentResponse[Any]]):
    """Results from a team execution."""

    def __init__(
        self, responses: list[AgentResponse[Any]], start_time: datetime | None = None
    ):
        super().__init__(responses)
        self.start_time = start_time or datetime.now()
        self.end_time = datetime.now()

    @property
    def duration(self) -> float:
        """Get execution duration in seconds."""
        return (self.end_time - self.start_time).total_seconds()

    @property
    def successful(self) -> list[AgentResponse[Any]]:
        """Get only successful responses."""
        return [r for r in self if r.success]

    @property
    def failed(self) -> list[AgentResponse[Any]]:
        """Get failed responses."""
        return [r for r in self if not r.success]

    def by_agent(self, name: str) -> AgentResponse[Any] | None:
        """Get response from specific agent."""
        return next((r for r in self if r.agent_name == name), None)

    def format_durations(self) -> str:
        """Format execution times."""
        parts = [f"{r.agent_name}: {r.timing:.2f}s" for r in self if r.timing is not None]
        return f"Individual times: {', '.join(parts)}\nTotal time: {self.duration:.2f}s"

    def to_chat_message(self) -> ChatMessage[str]:
        """Convert team response to a single chat message."""
        # Combine all responses into one structured message
        content = "\n\n".join(
            f"[{response.agent_name}]: {response.message.content}"
            for response in self
            if response.message
        )

        # Create a message that represents the group's output
        return ChatMessage(
            content=content,
            role="assistant",
            # Could include additional metadata about the group execution
            metadata={
                "type": "team_response",
                "agents": [r.agent_name for r in self],
                "duration": self.duration,
                "success_count": len(self.successful),
            },
        )


class Team[TDeps]:
    """Group of agents that can execute together."""

    def __init__(
        self,
        agents: list[AnyAgent[TDeps, Any]],
        *,
        shared_prompt: str | None = None,
        shared_deps: TDeps | None = None,
    ):
        self.agents = EventedList(agents)
        self.shared_prompt = shared_prompt
        self.shared_deps = shared_deps
        self.connections = TalkManager(self)
        self.team_talk = TeamTalk.from_agents(self.agents)

    def __rshift__(self, other: AnyAgent[Any, Any] | Team[Any] | str) -> TeamTalk:
        """Connect group to target agent(s).

        Returns:
            - list[Talk] when connecting to single agent (one Talk per source)
            - list[TeamTalk] when connecting to team (one TeamTalk per source)
        """
        return self.pass_results_to(other)

    def pass_results_to(self, other: AnyAgent[Any, Any] | Team[Any] | str) -> TeamTalk:
        match other:
            case str():
                if not self.agents[0].context.pool:
                    msg = "Pool required for forwarding to agent by name"
                    raise ValueError(msg)
                resolved = self.agents[0].context.pool.get_agent(other)
                return self.pass_results_to(resolved)
            case Team():
                return self.connections.connect_group_to(other)
            case _:
                return self.connections.connect_group_to(other)

    async def run_parallel(
        self, prompt: str | None = None, deps: TDeps | None = None
    ) -> TeamResponse:
        """Run all agents in parallel."""
        start_time = datetime.now()

        async def run_agent(agent: AnyAgent[TDeps, Any]) -> AgentResponse[Any]:
            try:
                start = time.perf_counter()
                actual_prompt = prompt or self.shared_prompt
                message = await agent.run(actual_prompt, deps=deps or self.shared_deps)
                timing = time.perf_counter() - start
                return AgentResponse(agent.name, message=message, timing=timing)
            except Exception as e:  # noqa: BLE001
                msg = ChatMessage(content="", role="assistant")
                return AgentResponse(agent_name=agent.name, message=msg, error=str(e))

        responses = await asyncio.gather(*[run_agent(a) for a in self.agents])
        return TeamResponse(responses, start_time)

    async def run_sequential(
        self,
        prompt: str | None = None,
        deps: TDeps | None = None,
    ) -> TeamResponse:
        """Run agents one after another."""
        start_time = datetime.now()
        results = []
        for agent in self.agents:
            try:
                start = time.perf_counter()
                message = await agent.run(
                    prompt or self.shared_prompt, deps=deps or self.shared_deps
                )
                timing = time.perf_counter() - start
                res = AgentResponse[str](
                    agent_name=agent.name, message=message, timing=timing
                )
                results.append(res)
            except Exception as e:  # noqa: BLE001
                msg = ChatMessage(content="", role="assistant")
                res = AgentResponse[str](agent_name=agent.name, message=msg, error=str(e))
                results.append(res)
        return TeamResponse(results, start_time)

    async def run_controlled(
        self,
        prompt: str | None = None,
        deps: TDeps | None = None,
        *,
        initial_agent: str | AnyAgent[TDeps, Any] | None = None,
        decision_callback: DecisionCallback = interactive_controller,
        router: AgentRouter | None = None,
    ) -> TeamResponse:
        results = []
        actual_prompt = prompt or self.shared_prompt
        actual_deps = deps or self.shared_deps
        start_time = datetime.now()

        # Resolve initial agent
        current_agent = (
            next(a for a in self.agents if a.name == initial_agent)
            if isinstance(initial_agent, str)
            else initial_agent or self.agents[0]
        )

        # Create router for decisions
        assert current_agent.context.pool
        router = router or CallbackRouter(current_agent.context.pool, decision_callback)
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

        return TeamResponse(results, start_time)

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
