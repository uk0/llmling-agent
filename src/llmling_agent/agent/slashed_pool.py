"""Pool-level slashed interface for agent control."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Self, overload

from psygnal import Signal
from pydantic import dataclasses
from slashed import BaseCommand, CommandStore, DefaultOutputWriter, OutputWriter

from llmling_agent.agent import Agent, AnyAgent, SlashedAgent
from llmling_agent.agent.slashed_agent import AgentOutput
from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from types import TracebackType

    from pydantic_ai.result import StreamedRunResult

    from llmling_agent.delegation.pool import AgentPool
    from llmling_agent.models.context import AgentContext
    from llmling_agent.models.messages import ChatMessage

logger = get_logger(__name__)


class _AgentResponseIterator:
    """Async iterator for agent responses."""

    def __init__(
        self,
        pool: SlashedPool[Any],
        content: str,
        *,
        agent: str | None = None,
        output: OutputWriter | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.pool = pool
        self.content = content
        self.target = agent
        self.output = output
        self.metadata = metadata
        self._tasks: dict[str, asyncio.Task[Any]] | None = None

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> ChatMessage[str]:
        # Initialize tasks on first iteration
        if self._tasks is None:
            if self.target:
                if self.target not in self.pool._slashed_agents:
                    msg = f"Agent {self.target} not found"
                    raise ValueError(msg)
                # Single agent - just run and stop iteration
                response = await self.pool._slashed_agents[self.target].run(
                    self.content,
                    output=self.output,
                    metadata={"sender": self.target, **(self.metadata or {})},
                )
                self._tasks = {}  # Mark as done
                return response

            # Multi-agent - create tasks
            self._tasks = {
                name: asyncio.create_task(
                    agent.run(
                        self.content,
                        output=self.output,
                        metadata={"sender": name, **(self.metadata or {})},
                    )
                )
                for name, agent in self.pool._slashed_agents.items()
            }

        # Stop if no more tasks
        if not self._tasks:
            raise StopAsyncIteration

        # Wait for next completed task
        done, _ = await asyncio.wait(
            self._tasks.values(), return_when=asyncio.FIRST_COMPLETED
        )
        for task in done:
            # Find and remove completed task
            name = next(n for n, t in self._tasks.items() if t == task)
            self._tasks.pop(name)
            return await task

        # Should never get here
        raise StopAsyncIteration


@dataclass
class MultiAgentResponse[T]:
    """Collected responses from multiple agents."""

    responses: dict[str, ChatMessage[T]]
    """Individual responses from each agent."""

    start_time: datetime
    """When the multi-agent call started."""

    end_time: datetime
    """When all responses were received."""

    @property
    def duration(self) -> float:
        """Total duration of all responses."""
        return (self.end_time - self.start_time).total_seconds()

    @property
    def combined_content(self) -> str:
        """All responses formatted with agent names."""
        return "\n\n".join(
            f"{name}: {msg.content}" for name, msg in self.responses.items()
        )


class SlashedPool[TDeps]:
    """Pool-level interface that routes messages to specific or all agents.

    Provides a unified interface for both single-agent and multi-agent operations.
    Messages can be routed using:
    - @ prefix: "@agent1 hello"
    - agent parameter: run("hello", agent="agent1")
    - broadcast: run("hello") -> sends to all agents
    """

    message_output = Signal(AgentOutput)
    streamed_output = Signal(AgentOutput)
    streaming_started = Signal(str, str)  # agent_name, message_id
    streaming_stopped = Signal(str, str)  # agent_name, message_id

    agent_added = Signal(str, Agent[Any])
    agent_removed = Signal(str)
    agent_command_executed = Signal(str, str)  # agent_name, command

    def __init__(
        self,
        pool: AgentPool,
        *,
        command_history_path: str | None = None,
    ):
        """Initialize pool interface.

        Args:
            pool: Agent pool to manage
            command_history_path: Optional path for command history
        """
        self.pool = pool
        self.commands = CommandStore(history_file=command_history_path)
        self._slashed_agents: dict[str, SlashedAgent[TDeps, Any]] = {}

        # Create SlashedAgent wrappers for each agent
        for name, agent in pool.agents.items():
            self._add_agent(name, agent)

    def _add_agent(self, name: str, agent: AnyAgent[TDeps, Any]):
        """Add a new slashed agent with signal forwarding."""
        slashed: SlashedAgent[TDeps, Any] = SlashedAgent(agent)
        # Forward all signals with agent name context
        slashed.message_output.connect(
            lambda output: self.message_output.emit(
                dataclasses.replace(output, metadata={"agent": name, **output.metadata})
            )
        )
        slashed.streamed_output.connect(
            lambda output: self.streamed_output.emit(
                dataclasses.replace(output, metadata={"agent": name, **output.metadata})
            )
        )
        slashed.streaming_started.connect(
            lambda msg_id: self.streaming_started.emit(name, msg_id)
        )
        slashed.streaming_stopped.connect(
            lambda msg_id: self.streaming_stopped.emit(name, msg_id)
        )
        self._slashed_agents[name] = slashed

    @overload
    async def run(
        self,
        content: str,
        *,
        agent: str,  # Specific agent
        output: OutputWriter | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ChatMessage[str]: ...

    @overload
    async def run(
        self,
        content: str,
        *,
        agent: None = None,  # Broadcasting
        output: OutputWriter | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MultiAgentResponse[str]: ...

    async def run(
        self,
        content: str,
        *,
        agent: str | None = None,
        output: OutputWriter | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ChatMessage[str] | MultiAgentResponse[str]:
        """Run with agent routing and wait for response(s).

        Args:
            content: Message content (can start with @agent_name)
            agent: Optional specific agent to target
            output: Optional output writer
            metadata: Additional metadata for messages

        Returns:
            - Single agent: ChatMessage with response
            - Multiple agents: MultiAgentResponse with all responses

        Raises:
            ValueError: If agent not found or @ syntax is invalid
        """
        if content.startswith("@"):
            parts = content[1:].split(maxsplit=1)
            if len(parts) != 2:  # noqa: PLR2004
                msg = "Usage: @agent_name message"
                raise ValueError(msg)
            target, message = parts
        else:
            target = agent  # type: ignore
            message = content

        # Single agent case
        if target:
            if target not in self._slashed_agents:
                msg = f"Agent {target} not found"
                raise ValueError(msg)
            return await self._slashed_agents[target].run(
                message, output=output, metadata={"sender": target, **(metadata or {})}
            )

        # Multi-agent case
        start_time = datetime.now()
        tasks = [
            agent.run(
                message, output=output, metadata={"sender": name, **(metadata or {})}
            )
            for name, agent in self._slashed_agents.items()
        ]
        results = await asyncio.gather(*tasks)
        responses = dict(zip(self._slashed_agents.keys(), results))
        return MultiAgentResponse(
            responses, start_time=start_time, end_time=datetime.now()
        )

    @asynccontextmanager
    async def run_stream(
        self,
        content: str,
        *,
        agent: str,
        output: OutputWriter | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamedRunResult[AgentContext[TDeps], str]]:
        """Stream responses from a specific agent.

        Args:
            content: Message to send
            agent: Name of agent to stream from
            output: Optional output writer
            metadata: Additional metadata for messages

        Yields:
            Stream of ChatMessages from the agent

        Raises:
            ValueError: If agent not found
        """
        if agent not in self._slashed_agents:
            msg = f"Agent {agent} not found"
            raise ValueError(msg)

        metadata = {"sender": agent, **(metadata or {})}
        async with self._slashed_agents[agent].run_stream(
            content,
            output=output,
            metadata=metadata,
        ) as stream:
            yield stream

    async def run_iter(
        self,
        content: str,
        *,
        agent: str | None = None,
        output: OutputWriter | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncIterator[ChatMessage[str]]:
        """Run and yield responses as they complete.

        Args:
            content: Message content (can start with @agent_name)
            agent: Optional specific agent to target
            output: Optional output writer
            metadata: Additional metadata

        Returns:
            AsyncIterator yielding ChatMessages as they complete

        Raises:
            ValueError: If agent not found or @ syntax is invalid
        """
        # Parse @ syntax if present
        if content.startswith("@"):
            parts = content[1:].split(maxsplit=1)
            if len(parts) != 2:  # noqa: PLR2004
                msg = "Usage: @agent_name message"
                raise ValueError(msg)
            target, message = parts
        else:
            target = agent  # type: ignore
            message = content

        # Return properly implemented iterator
        return _AgentResponseIterator(
            self,
            message,
            agent=target,
            output=output,
            metadata=metadata,
        )

    async def execute_command(
        self,
        command: str,
        output: OutputWriter | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Execute command with optional agent routing.

        Args:
            command: Command to execute (can start with @agent_name)
            output: Optional output writer
            metadata: Additional metadata

        Returns:
            Command result message
        """
        output = output or DefaultOutputWriter()
        ctx = self.commands.create_context(self, output_writer=output, metadata=metadata)

        # Handle agent-specific commands
        if command.startswith("@"):
            parts = command[1:].split(maxsplit=1)
            if len(parts) != 2:  # noqa: PLR2004
                return "Usage: @agent_name /command"

            agent_name, agent_command = parts
            if agent_name not in self._slashed_agents:
                return f"Agent {agent_name} not found"

            # Forward to specific agent
            slashed_agent = self._slashed_agents[agent_name]
            result = await slashed_agent.handle_command(
                agent_command,
                output=output,
                metadata=metadata,
            )
            self.agent_command_executed.emit(agent_name, agent_command)
            return str(result.content)

        # Pool-level command
        await self.commands.execute_command(command, ctx)
        return "finished"

    def register_pool_command(self, command: BaseCommand):
        """Register a pool-level command."""
        self.commands.register_command(command)

    @property
    def agents(self) -> dict[str, SlashedAgent[TDeps, Any]]:
        """Access to all slashed agents."""
        return self._slashed_agents.copy()

    async def __aenter__(self) -> Self:
        """Enter async context."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit async context."""
        # Cleanup will be handled by pool
