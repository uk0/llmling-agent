from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, overload

from psygnal.containers import EventedList
from typing_extensions import TypeVar

from llmling_agent.agent.connection import ConnectionManager
from llmling_agent.delegation.teamrun import ExecutionMode, TeamRun
from llmling_agent.log import get_logger
from llmling_agent.talk import QueueStrategy, TeamTalk
from llmling_agent.utils.inspection import has_return_type
from llmling_agent.utils.tasks import TaskManagerMixin
from llmling_agent_providers.base import StreamingResponseProtocol


logger = get_logger(__name__)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from datetime import timedelta
    import os

    import PIL.Image
    from tokonomics.pydanticai_cost import Usage
    from toprompt import AnyPromptType

    from llmling_agent.agent import AnyAgent
    from llmling_agent.common_types import AgentName, AnyTransformFn, AsyncFilterFn
    from llmling_agent.models.forward_targets import ConnectionType
    from llmling_agent.models.messages import ChatMessage, TeamResponse
    from llmling_agent.models.providers import ProcessorCallback


TDeps = TypeVar("TDeps", default=None)


class Team[TDeps](TaskManagerMixin):
    """Group of agents that can execute together."""

    def __init__(
        self,
        agents: Sequence[AnyAgent[TDeps, Any]],
        *,
        shared_prompt: str | None = None,
        name: str | None = None,
    ):
        self.agents = EventedList(list(agents))
        self.shared_prompt = shared_prompt
        self.name = name or " & ".join([i.name for i in agents])
        self.connections = ConnectionManager(self)
        self.team_talk = TeamTalk.from_agents(self.agents)

    def __repr__(self) -> str:
        """Create a readable representation of the team."""
        members = ", ".join(agent.name for agent in self.agents)
        name = f" ({self.name})" if self.name else ""
        return f"Team[{len(self.agents)}]{name}: {members}"

    def __contains__(self, item: Any) -> bool:
        """Check if an agent is part of the team."""
        return item in self.agents

    def __iter__(self):
        """Iterate over team members."""
        return iter(self.agents)

    def __len__(self) -> int:
        """Get number of team members."""
        return len(self.agents)

    def __getitem__(self, index_or_name: int | AgentName) -> AnyAgent[TDeps, Any]:
        """Get team member by index or name."""
        if isinstance(index_or_name, str):
            return next(agent for agent in self.agents if agent.name == index_or_name)
        return self.agents[index_or_name]

    @overload
    def __and__(self, other: Team[None]) -> Team[None]: ...

    @overload
    def __and__(self, other: Team[TDeps]) -> Team[TDeps]: ...

    @overload
    def __and__(self, other: Team[Any]) -> Team[Any]: ...

    @overload
    def __and__(self, other: AnyAgent[TDeps, Any]) -> Team[TDeps]: ...

    @overload
    def __and__(self, other: AnyAgent[Any, Any]) -> Team[Any]: ...

    def __and__(self, other: Team[Any] | AnyAgent[Any, Any]) -> Team[Any]:
        """Combine teams, preserving type safety for same types."""
        # Combine agents

        match other:
            case Team():
                combined_agents = [*self.agents, *other.agents]
                combined_prompts = []
                if self.shared_prompt:
                    combined_prompts.append(self.shared_prompt)
                if other.shared_prompt:
                    combined_prompts.append(other.shared_prompt)
                shared = "\n".join(combined_prompts) if combined_prompts else None
                return Team(agents=combined_agents, shared_prompt=shared)
            case _:  # AnyAgent case
                # Keep same deps if types match
                agents = [*self.agents, other]
                return Team(agents=agents, shared_prompt=self.shared_prompt)

    def __or__(
        self, other: AnyAgent[Any, Any] | ProcessorCallback[Any] | Team[Any]
    ) -> TeamRun[TDeps]:
        """Create a pipeline using | operator.

        Example:
            pipeline = team | transform | other_team  # Sequential processing
        """
        from llmling_agent.agent import Agent, StructuredAgent

        match other:
            case Team():
                # Create sequential execution with all agents
                team = Team([*self.agents, *other.agents])
                execution = TeamRun(team, mode="sequential")
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

    def __rshift__(
        self, other: AnyAgent[Any, Any] | Team[Any] | ProcessorCallback[Any]
    ) -> TeamTalk:
        """Connect group to target agent(s).

        Returns:
            - list[Talk] when connecting to single agent (one Talk per source)
            - list[TeamTalk] when connecting to team (one TeamTalk per source)
        """
        return self.pass_results_to(other)

    def pass_results_to(
        self,
        other: AnyAgent[Any, Any] | Team[Any] | ProcessorCallback[Any],
        *,
        connection_type: ConnectionType = "run",
        priority: int = 0,
        delay: timedelta | None = None,
        queued: bool = False,
        queue_strategy: QueueStrategy = "latest",
        transform: AnyTransformFn | None = None,
        filter_condition: AsyncFilterFn | None = None,
        stop_condition: AsyncFilterFn | None = None,
        exit_condition: AsyncFilterFn | None = None,
    ) -> TeamTalk:
        """Forward results to another agent or all agents in a team."""
        from llmling_agent.agent import Agent, StructuredAgent

        if callable(other):
            if has_return_type(other, str):
                other = Agent.from_callback(other)
            else:
                other = StructuredAgent.from_callback(other)
        return self.connections.connect_group_to(
            other,
            connection_type=connection_type,
            priority=priority,
            delay=delay,
            queued=queued,
            queue_strategy=queue_strategy,
            transform=transform,
            filter_condition=filter_condition,
            stop_condition=stop_condition,
            exit_condition=exit_condition,
        )

    def monitored(self, mode: ExecutionMode) -> TeamRun[TDeps]:
        """Create a monitored execution."""
        return TeamRun(self, mode)

    async def run_parallel(
        self,
        *prompts: AnyPromptType | PIL.Image.Image | os.PathLike[str] | None,
    ) -> TeamResponse:
        """Run all agents in parallel."""
        execution = TeamRun(self, "parallel")
        return await execution.run(*prompts)

    async def run_sequential(
        self,
        *prompts: AnyPromptType | PIL.Image.Image | os.PathLike[str] | None,
    ) -> TeamResponse:
        """Run agents one after another."""
        execution = TeamRun(self, "sequential")
        return await execution.run(*prompts)

    async def chain(
        self,
        *prompts: AnyPromptType | PIL.Image.Image | os.PathLike[str] | None,
        require_all: bool = True,
    ) -> ChatMessage:
        """Pass message through the chain of team members.

        Each agent processes the result of the previous one.

        Args:
            prompts: Initial messages to process
            require_all: If True, all agents must succeed

        Returns:
            Final processed message

        Raises:
            ValueError: If chain breaks and require_all=True
        """
        current_message = prompts

        for agent in self.agents:
            try:
                result = await agent.run(*current_message)
                current_message = (result.content,)
            except Exception as e:
                if require_all:
                    msg = f"Chain broken at {agent.name}: {e}"
                    raise ValueError(msg) from e
                logger.warning("Chain handler %s failed: %s", agent.name, e)

        return result

    @asynccontextmanager
    async def chain_stream(
        self,
        *prompts: AnyPromptType | PIL.Image.Image | os.PathLike[str] | None,
        require_all: bool = True,
    ) -> AsyncIterator[StreamingResponseProtocol]:
        """Stream results through chain of team members."""
        async with AsyncExitStack() as stack:
            streams: list[StreamingResponseProtocol[str]] = []
            current_message = prompts

            # Set up all streams
            for agent in self.agents:
                try:
                    stream = await stack.enter_async_context(
                        agent.run_stream(*current_message)
                    )
                    streams.append(stream)
                    # Wait for complete response for next agent
                    async for chunk in stream.stream():
                        current_message = chunk
                        if stream.is_complete:
                            current_message = (stream.formatted_content,)  # type: ignore
                            break
                except Exception as e:
                    if require_all:
                        msg = f"Chain broken at {agent.name}: {e}"
                        raise ValueError(msg) from e
                    logger.warning("Chain handler %s failed: %s", agent.name, e)

            # Create a stream-like interface for the chain
            class ChainStream(StreamingResponseProtocol[str]):
                def __init__(self):
                    self.streams = streams
                    self.current_stream_idx = 0
                    self.is_complete = False
                    self.model_name = None

                def usage(self) -> Usage:
                    @dataclass
                    class Usage:
                        total_tokens: int | None
                        request_tokens: int | None
                        response_tokens: int | None

                    return Usage(0, 0, 0)

                async def stream(self) -> AsyncIterator[str]:  # type: ignore
                    for idx, stream in enumerate(self.streams):
                        self.current_stream_idx = idx
                        async for chunk in stream.stream():
                            yield chunk
                            if idx == len(self.streams) - 1 and stream.is_complete:
                                self.is_complete = True

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
