from __future__ import annotations

from collections.abc import Callable
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal, overload

from psygnal.containers import EventedList
from pydantic_ai.result import StreamedRunResult
from typing_extensions import TypeVar

from llmling_agent.agent.connection import TalkManager, TeamTalk
from llmling_agent.delegation.execution import TeamRun
from llmling_agent.delegation.pool import AgentResponse
from llmling_agent.log import get_logger
from llmling_agent.models.messages import ChatMessage
from llmling_agent.utils.tasks import TaskManagerMixin


logger = get_logger(__name__)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from llmling_agent.agent import AnyAgent
    from llmling_agent.delegation.callbacks import DecisionCallback
    from llmling_agent.delegation.router import AgentRouter
    from llmling_agent.models.context import AgentContext
    from llmling_agent.models.forward_targets import ConnectionType


TDeps = TypeVar("TDeps", default=None)
ResultData = TypeVar("ResultData", default=str)


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


class Team[TDeps](TaskManagerMixin):
    """Group of agents that can execute together."""

    def __init__(
        self,
        agents: list[AnyAgent[TDeps, Any]],
        *,
        shared_prompt: str | None = None,
        shared_deps: TDeps | None = None,
        name: str | None = None,
    ):
        self.agents = EventedList(agents)
        self.shared_prompt = shared_prompt
        self.shared_deps = shared_deps
        self.name = name or " & ".join([i.name for i in agents])
        self.connections = TalkManager(self)
        self.team_talk = TeamTalk.from_agents(self.agents)

    @overload
    def __and__(self, other: Team[None]) -> Team[None]: ...

    @overload
    def __and__(self, other: Team[TDeps]) -> Team[list[TDeps]]: ...

    @overload
    def __and__(self, other: Team[Any]) -> Team[list[Any]]: ...

    @overload
    def __and__(self, other: AnyAgent[TDeps, Any]) -> Team[TDeps]: ...

    @overload
    def __and__(self, other: AnyAgent[Any, Any]) -> Team[list[Any]]: ...

    def __and__(self, other: Team[Any] | AnyAgent[Any, Any]) -> Team[Any]:
        """Combine teams, preserving type safety for same types."""
        # Combine agents
        from llmling_agent.agent import Agent, StructuredAgent

        match other:
            case Team():
                # Combine agents
                combined_agents = [*self.agents, *other.agents]

                # Handle deps
                if self.shared_deps is None and other.shared_deps is None:
                    combined_deps = None
                else:
                    combined_deps = []
                    if self.shared_deps is not None:
                        combined_deps.append(self.shared_deps)
                    if other.shared_deps is not None:
                        combined_deps.append(other.shared_deps)

                # Combine prompts with line break
                combined_prompts = []
                if self.shared_prompt:
                    combined_prompts.append(self.shared_prompt)
                if other.shared_prompt:
                    combined_prompts.append(other.shared_prompt)

                return Team(
                    agents=combined_agents,
                    shared_deps=combined_deps,
                    shared_prompt="\n".join(combined_prompts)
                    if combined_prompts
                    else None,
                )
            case _:  # AnyAgent case
                # Keep same deps if types match
                deps = (
                    self.shared_deps
                    if isinstance(other, Agent | StructuredAgent)
                    and other.context.data == self.shared_deps
                    else [self.shared_deps]
                    if self.shared_deps is not None
                    else None
                )
                return Team(
                    agents=[*self.agents, other],
                    shared_deps=deps,
                    shared_prompt=self.shared_prompt,
                )

    def __or__(self, other: AnyAgent[Any, Any] | Callable | Team[Any]) -> TeamRun[TDeps]:
        """Create a pipeline using | operator.

        Example:
            pipeline = team | transform | other_team  # Sequential processing
        """
        from llmling_agent.agent import Agent, StructuredAgent

        match other:
            case Team():
                # Create sequential execution with all agents
                execution = TeamRun(
                    Team([*self.agents, *other.agents]), mode="sequential"
                )
            case Callable():
                # Convert callable to agent and add to pipeline
                from llmling_agent import Agent
                from llmling_agent_providers.callback import CallbackProvider

                provider = CallbackProvider(other)
                new_agent = Agent(provider=provider, name=other.__name__)
                execution = TeamRun(Team([*self.agents, new_agent]), mode="sequential")
            case Agent() | StructuredAgent():  # Agent case
                execution = TeamRun(Team([*self.agents, other]), mode="sequential")
            case _:
                msg = f"Invalid pipeline element: {other}"
                raise ValueError(msg)

        # Setup connections for sequential processing
        for i in range(len(execution.team.agents) - 1):
            current = execution.team.agents[i]
            next_agent = execution.team.agents[i + 1]
            current.pass_results_to(next_agent)

        return execution

    def __rshift__(self, other: AnyAgent[Any, Any] | Team[Any] | str) -> TeamTalk:
        """Connect group to target agent(s).

        Returns:
            - list[Talk] when connecting to single agent (one Talk per source)
            - list[TeamTalk] when connecting to team (one TeamTalk per source)
        """
        return self.pass_results_to(other)

    def pass_results_to(
        self,
        other: AnyAgent[Any, Any] | Team[Any] | str,
        connection_type: ConnectionType = "run",
        priority: int = 0,
        delay: timedelta | None = None,
    ) -> TeamTalk:
        match other:
            case str():
                if not self.agents[0].context.pool:
                    msg = "Pool required for forwarding to agent by name"
                    raise ValueError(msg)
                resolved = self.agents[0].context.pool.get_agent(other)
                return self.pass_results_to(
                    resolved,
                    priority=priority,
                    delay=delay,
                    connection_type=connection_type,
                )
            case _:
                return self.connections.connect_group_to(
                    other,
                    connection_type=connection_type,
                    priority=priority,
                    delay=delay,
                )

    def monitored(
        self,
        mode: Literal["parallel", "sequential", "controlled"],
    ) -> TeamRun[TDeps]:
        """Create a monitored execution."""
        return TeamRun(self, mode)

    async def run_parallel(
        self,
        prompt: str | None = None,
        deps: TDeps | None = None,
    ) -> TeamResponse:
        """Run all agents in parallel."""
        execution = TeamRun(self, "parallel")
        return await execution.run(prompt, deps)

    async def run_sequential(
        self,
        prompt: str | None = None,
        deps: TDeps | None = None,
    ) -> TeamResponse:
        """Run agents one after another."""
        execution = TeamRun(self, "sequential")
        return await execution.run(prompt, deps)

    async def run_controlled(
        self,
        prompt: str | None = None,
        deps: TDeps | None = None,
        *,
        initial_agent: str | AnyAgent[TDeps, Any] | None = None,
        decision_callback: DecisionCallback | None = None,
        router: AgentRouter | None = None,
    ) -> TeamResponse:
        execution = TeamRun(self, "controlled")
        return await execution.run(prompt, deps)

    async def chain(
        self,
        message: Any,
        *,
        require_all: bool = True,
    ) -> ChatMessage:
        """Pass message through the chain of team members.

        Each agent processes the result of the previous one.

        Args:
            message: Initial message to process
            require_all: If True, all agents must succeed

        Returns:
            Final processed message

        Raises:
            ValueError: If chain breaks and require_all=True
        """
        current_message = message

        for agent in self.agents:
            try:
                result = await agent.run(current_message)
                current_message = result.content
            except Exception as e:
                if require_all:
                    msg = f"Chain broken at {agent.name}: {e}"
                    raise ValueError(msg) from e
                logger.warning("Chain handler %s failed: %s", agent.name, e)

        return result

    @asynccontextmanager
    async def chain_stream(
        self,
        message: Any,
        *,
        require_all: bool = True,
    ) -> AsyncIterator[StreamedRunResult[AgentContext[TDeps], str]]:
        """Stream results through chain of team members."""
        from llmling_agent.models.context import AgentContext

        async with AsyncExitStack() as stack:
            streams: list[StreamedRunResult[AgentContext[TDeps], str]] = []
            current_message = message

            # Set up all streams
            for agent in self.agents:
                try:
                    stream = await stack.enter_async_context(
                        agent.run_stream(current_message)
                    )
                    streams.append(stream)
                    # Wait for complete response for next agent
                    async for chunk in stream.stream():
                        current_message = chunk
                        if stream.is_complete:
                            current_message = stream.formatted_content  # type: ignore
                            break
                except Exception as e:
                    if require_all:
                        msg = f"Chain broken at {agent.name}: {e}"
                        raise ValueError(msg) from e
                    logger.warning("Chain handler %s failed: %s", agent.name, e)

            # Create a stream-like interface for the chain
            class ChainStream(StreamedRunResult[AgentContext[TDeps], str]):
                def __init__(self):
                    self.streams = streams
                    self.current_stream_idx = 0
                    self.is_final = False

                async def stream(self) -> AsyncIterator[str]:  # type: ignore
                    for idx, stream in enumerate(self.streams):
                        self.current_stream_idx = idx
                        async for chunk in stream.stream():
                            yield chunk
                            if idx == len(self.streams) - 1 and stream.is_complete:
                                self.is_final = True

            yield ChainStream()

    async def distribute(
        self,
        content: str,
        *,
        tools: list[str] | None = None,
        resources: list[str] | None = None,
    ) -> None:
        """Distribute content and capabilities to all team members.

        This method provides content and optional capabilities to every agent in the team.

        Args:
            content: Text content to add as context for all agents
            tools: Optional list of tools to register with each agent.
                  Can be import paths or callables.
            resources: Optional list of resources to load into each agent's context.
                      Can be paths, URLs, or resource configurations.

        """
        for agent in self.agents:
            # Add context message
            agent.conversation.add_context_message(content, source="distribution")

            # Register tools if provided
            if tools:
                for tool in tools:
                    agent.tools.register_tool(tool)

            # Load resources if provided
            if resources:
                for resource in resources:
                    await agent.conversation.load_context_source(resource)
