"""Team execution management and monitoring."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
import inspect
from time import perf_counter
from typing import TYPE_CHECKING, Any, Literal

from llmling_agent.delegation.controllers import interactive_controller
from llmling_agent.delegation.pool import AgentResponse
from llmling_agent.delegation.router import (
    AgentRouter,
    AwaitResponseDecision,
    CallbackRouter,
    EndDecision,
    RouteDecision,
)
from llmling_agent.log import get_logger
from llmling_agent.models.messages import ChatMessage
from llmling_agent.utils.tasks import TaskManagerMixin


if TYPE_CHECKING:
    from psygnal import SignalInstance

    from llmling_agent.agent import AnyAgent
    from llmling_agent.agent.agent import Agent
    from llmling_agent.delegation import DecisionCallback, Team, TeamResponse
    from llmling_agent.models.agents import ToolCallInfo


logger = get_logger(__name__)

ExecutionMode = Literal["parallel", "sequential", "controlled"]
"""The execution mode for a TeamRun."""


@dataclass(frozen=True)
class TeamRunStats:
    """Statistics about a team execution."""

    start_time: datetime
    received_messages: dict[str, list[ChatMessage]]  # agent -> messages
    sent_messages: dict[str, list[ChatMessage]]  # agent -> messages
    tool_calls: dict[str, list[ToolCallInfo]]  # agent -> tool calls
    error_log: list[tuple[str, str, datetime]]  # (agent, error, timestamp)
    duration: float

    @property
    def active_agents(self) -> list[str]:
        """Get currently active agents."""
        return [
            name
            for name in self.received_messages
            if len(self.received_messages[name]) > len(self.sent_messages[name])
        ]

    @property
    def message_counts(self) -> dict[str, int]:
        """Get message count per agent."""
        return {name: len(messages) for name, messages in self.sent_messages.items()}

    @property
    def total_tokens(self) -> int:
        """Get total token usage across all agents."""
        return sum(
            msg.cost_info.token_usage["total"]
            for messages in self.sent_messages.values()
            for msg in messages
            if msg.cost_info
        )

    @property
    def total_cost(self) -> float:
        """Get total cost across all agents."""
        return sum(
            float(msg.cost_info.total_cost)
            for messages in self.sent_messages.values()
            for msg in messages
            if msg.cost_info
        )

    @property
    def tool_counts(self) -> dict[str, int]:
        """Get tool usage count per agent."""
        return {name: len(calls) for name, calls in self.tool_calls.items()}

    @property
    def is_active(self) -> bool:
        """Whether any agents are still processing."""
        return bool(self.active_agents)

    @property
    def has_errors(self) -> bool:
        """Whether any errors occurred."""
        return bool(self.error_log)


class TeamRunMonitor:
    """Monitors team execution through agent signals."""

    def __init__(self, team: Team):
        self.team = team
        self.start_time = datetime.now()
        self._stop_event = asyncio.Event()

        # Raw event collection
        self._received_messages: dict[str, list[ChatMessage]] = {
            agent.name: [] for agent in team.agents
        }
        self._sent_messages: dict[str, list[ChatMessage]] = {
            agent.name: [] for agent in team.agents
        }
        self._errors: dict[str, list[tuple[str, datetime]]] = {
            agent.name: [] for agent in team.agents
        }
        self._tool_calls: dict[str, list[ToolCallInfo]] = {
            agent.name: [] for agent in team.agents
        }

        # Track signal connections for cleanup
        self._signal_connections: list[tuple[SignalInstance, Callable]] = []

    def start(self):
        """Start monitoring."""
        # Connect signals for all agents
        logger.debug("TeamRunMonitor starting")
        self._stop_event.clear()
        for agent in self.team.agents:
            # Message tracking
            self._signal_connections.extend([
                (
                    agent.message_received,
                    lambda msg, name=agent.name: self._received_messages[name].append(
                        msg
                    ),
                ),
                (
                    agent.message_sent,
                    lambda msg, name=agent.name: self._sent_messages[name].append(msg),
                ),
                (
                    agent.run_failed,
                    lambda msg, exc, name=agent.name: self._errors[name].append((
                        str(exc),
                        datetime.now(),
                    )),
                ),
                (
                    agent.tool_used,
                    lambda info, name=agent.name: self._tool_calls[name].append(info),
                ),
            ])

        # Connect all tracked signals
        for signal, handler in self._signal_connections:
            signal.connect(handler)

    def stop(self):
        """Stop monitoring and cleanup signals."""
        logger.debug("TeamRunMonitor stopping")
        self._stop_event.set()
        for signal, handler in self._signal_connections:
            signal.disconnect(handler)
        self._signal_connections.clear()

    @property
    def stats(self) -> TeamRunStats:
        """Get current execution statistics."""
        log = [(n, err, ts) for n, errors in self._errors.items() for err, ts in errors]
        return TeamRunStats(
            start_time=self.start_time,
            received_messages=self._received_messages,
            sent_messages=self._sent_messages,
            tool_calls=self._tool_calls,
            error_log=log,
            duration=(datetime.now() - self.start_time).total_seconds(),
        )


class TeamRun[TDeps](TaskManagerMixin):
    """Handles team operations with optional monitoring."""

    def __init__(
        self,
        team: Team[TDeps],
        mode: ExecutionMode,
    ):
        super().__init__()
        self.team = team
        self.mode = mode
        self._monitor: TeamRunMonitor | None = None
        self._main_task: asyncio.Task[TeamResponse] | None = None

    def __or__(self, other: Agent | Callable | Team | TeamRun) -> TeamRun:
        from llmling_agent import Agent, Team
        from llmling_agent_providers.callback import CallbackProvider

        match other:
            case Agent():
                self.team.agents.append(other)
            case Callable():
                provider = CallbackProvider(other)
                position = len(self.team.agents) + 1
                name = f"{other.__name__}_{position}"
                new_agent = Agent(provider=provider, name=name)
                self.team.agents.append(new_agent)
            case Team():
                # Flatten team
                self.team.agents.extend(other.agents)
            case TeamRun():
                # Merge executions
                self.team.agents.extend(other.team.agents)
        return self

    async def start(
        self,
        prompt: str | None = None,
        deps: Any | None = None,
        monitor_callback: Callable[[TeamRunStats], Any] | None = None,
        monitor_interval: float = 0.1,
        **kwargs: Any,
    ) -> TeamResponse:
        """Start execution with optional monitoring."""
        self._monitor = TeamRunMonitor(self.team)
        self._monitor.start()

        if monitor_callback:

            async def _monitor():
                logger.debug("Monitor task starting")
                assert self._monitor
                while not self._monitor._stop_event.is_set():
                    logger.debug("Monitor checking stats")
                    stats = self.stats
                    logger.debug("Stats: active_agents=%s", stats.active_agents)
                    if inspect.iscoroutinefunction(monitor_callback):
                        await monitor_callback(stats)
                    else:
                        monitor_callback(stats)
                    await asyncio.sleep(monitor_interval)
                logger.debug("Monitor task stopping")

            self.create_task(_monitor(), name="stats_monitor")

        return await self._execute(prompt, **kwargs)

    def start_background(
        self,
        prompt: str | None = None,
        deps: TDeps | None = None,
        monitor_callback: Callable[[TeamRunStats], Any] | None = None,
        monitor_interval: float = 0.1,
        **kwargs: Any,
    ) -> None:
        if self._main_task:
            msg = "Execution already running"
            raise RuntimeError(msg)
        coro = self.start(
            prompt,
            deps,
            monitor_callback=monitor_callback,
            monitor_interval=monitor_interval,
            **kwargs,
        )
        self._main_task = self.create_task(coro, name="main_execution")

    def monitor(
        self,
        callback: Callable[[TeamRunStats], Any],
        interval: float = 0.1,
    ) -> None:
        """Monitor execution with callback.

        Args:
            callback: Function to call with stats updates
            interval: How often to check for updates (seconds)
        """

        async def _monitor():
            while self.is_running:
                # Direct callback with stats
                if inspect.iscoroutinefunction(callback):
                    await callback(self.stats)
                else:
                    callback(self.stats)
                await asyncio.sleep(interval)

        self.create_task(_monitor(), name="stats_monitor")

    @property
    def is_running(self) -> bool:
        """Whether execution is currently running."""
        return bool(self._main_task and not self._main_task.done())

    async def wait(self) -> TeamResponse:
        if not self._main_task:
            msg = "No execution running"
            raise RuntimeError(msg)
        try:
            return await self._main_task
        finally:
            if self._monitor:
                self._monitor.stop()
            await self.cleanup_tasks()

    async def cancel(self) -> None:
        """Cancel execution and cleanup."""
        if self._main_task:
            self._main_task.cancel()
        await self.cleanup_tasks()

    async def run(
        self,
        prompt: str | None = None,
        deps: TDeps | None = None,
        **kwargs: Any,
    ) -> TeamResponse:
        """Execute directly without monitoring."""
        return await self._execute(prompt, **kwargs)

    async def _execute(self, prompt: str | None = None, **kwargs: Any) -> TeamResponse:
        """Common execution logic."""
        self._monitor = TeamRunMonitor(self.team)
        self._monitor.start()
        try:
            match self.mode:
                case "parallel":
                    return await self._run_parallel(prompt)
                case "sequential":
                    return await self._run_sequential(prompt)
                case "controlled":
                    return await self._run_controlled(prompt, **kwargs)
                case _:
                    msg = f"Invalid mode: {self.mode}"
                    raise ValueError(msg)
        finally:
            self._monitor.stop()

    @property
    def stats(self) -> TeamRunStats:
        """Get current execution statistics."""
        if not self._monitor:
            # Return empty stats if not monitoring
            return TeamRunStats(
                start_time=datetime.now(),
                received_messages={agent.name: [] for agent in self.team.agents},
                sent_messages={agent.name: [] for agent in self.team.agents},
                tool_calls={agent.name: [] for agent in self.team.agents},
                error_log=[],
                duration=0.0,
            )
        return self._monitor.stats

    async def _run_parallel(self, prompt: str | None = None) -> TeamResponse:
        """Execute in parallel mode.

        All agents run simultaneously and independently.
        """
        from llmling_agent.delegation.agentgroup import TeamResponse

        start_time = datetime.now()

        # Combine shared prompt with user prompt if both exist
        final_prompt = None
        if self.team.shared_prompt and prompt:
            final_prompt = f"{self.team.shared_prompt}\n\n{prompt}"
        else:
            final_prompt = self.team.shared_prompt or prompt

        async def run_agent(agent: AnyAgent[TDeps, Any]) -> AgentResponse[Any]:
            try:
                start = perf_counter()
                message = await agent.run(final_prompt)
                timing = perf_counter() - start
                return AgentResponse(agent.name, message=message, timing=timing)
            except Exception as e:  # noqa: BLE001
                msg = ChatMessage(content="", role="assistant")
                return AgentResponse(agent_name=agent.name, message=msg, error=str(e))

        responses = await asyncio.gather(*[run_agent(a) for a in self.team.agents])
        return TeamResponse(responses, start_time)

    async def _run_sequential(self, prompt: str | None = None) -> TeamResponse:
        """Execute in sequential mode.

        Agents run one after another, in order.
        """
        from llmling_agent.delegation.agentgroup import TeamResponse

        start_time = datetime.now()
        results = []
        # Combine shared prompt with user prompt if both exist
        final_prompt = None
        if self.team.shared_prompt and prompt:
            final_prompt = f"{self.team.shared_prompt}\n\n{prompt}"
        else:
            final_prompt = self.team.shared_prompt or prompt
        current_input = final_prompt
        for agent in self.team.agents:
            try:
                start = perf_counter()
                message = await agent.run(current_input)
                current_input = str(message.data)
                timing = perf_counter() - start
                res = AgentResponse[str](
                    agent_name=agent.name, message=message, timing=timing
                )
                results.append(res)
            except Exception as e:  # noqa: BLE001
                msg = ChatMessage(content="", role="assistant")
                res = AgentResponse[str](agent_name=agent.name, message=msg, error=str(e))
                results.append(res)

        return TeamResponse(results, start_time)

    async def _run_controlled(
        self,
        prompt: str | None = None,
        *,
        initial_agent: str | AnyAgent[TDeps, Any] | None = None,
        decision_callback: DecisionCallback | None = None,
        router: AgentRouter | None = None,
        **kwargs: Any,
    ) -> TeamResponse:
        """Execute in controlled mode.

        Execution flow is controlled by routing decisions.
        """
        from llmling_agent.delegation.agentgroup import TeamResponse

        # Combine shared prompt with user prompt if both exist
        final_prompt = None
        if self.team.shared_prompt and prompt:
            final_prompt = f"{self.team.shared_prompt}\n\n{prompt}"
        else:
            final_prompt = self.team.shared_prompt or prompt

        results = []
        start_time = datetime.now()
        if not decision_callback:
            decision_callback = interactive_controller
        # Resolve initial agent
        current_agent = (
            next(a for a in self.team.agents if a.name == initial_agent)
            if isinstance(initial_agent, str)
            else initial_agent or self.team.agents[0]
        )

        # Create router for decisions
        assert current_agent.context.pool
        router = router or CallbackRouter(current_agent.context.pool, decision_callback)
        current_message = final_prompt

        while True:
            # Get response from current agent
            now = perf_counter()
            message = await current_agent.run(current_message)
            duration = perf_counter() - now
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
                        (a for a in self.team.agents if a.name == decision.target_agent),
                        current_agent,
                    )
                    current_message = str(response.message.content)

        return TeamResponse(results, start_time)


if __name__ == "__main__":
    import asyncio

    from llmling_agent.delegation import AgentPool

    async def on_stats_update(stats: TeamRunStats):
        """Handle stats updates."""
        print(
            f"\rActive: {stats.active_agents} | Messages: {stats.message_counts}", end=""
        )

    async def main():
        async with AgentPool[None]() as pool:
            analyzer = await pool.add_agent(
                "analyzer",
                system_prompt="You analyze text in a formal way.",
                model="openai:gpt-4o-mini",
            )
            summarizer = await pool.add_agent(
                "summarizer",
                system_prompt="You create concise summaries.",
                model="openai:gpt-4o-mini",
            )

            team = pool.create_team([analyzer, summarizer])
            text = "The quick brown fox jumps over the lazy dog."

            print("\n=== Monitored Parallel Execution ===")
            execution = team.monitored("parallel")

            # Start execution and monitoring
            execution.start_background(text)
            execution.monitor(on_stats_update)

            # Wait for completion
            response = await execution.wait()
            print(response)
            print("\n\nFinal Stats:")
            print(f"Duration: {execution.stats.duration:.2f}s")
            print(f"Total tokens: {execution.stats.total_tokens}")
            print(f"Total cost: ${execution.stats.total_cost:.4f}")

    asyncio.run(main())
