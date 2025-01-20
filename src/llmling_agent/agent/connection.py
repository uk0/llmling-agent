"""Manages message flow between agents/groups."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal, Self, overload

from psygnal import Signal
from typing_extensions import TypeVar

from llmling_agent.log import get_logger
from llmling_agent.models.messages import ChatMessage, FormatStyle


if TYPE_CHECKING:
    from toprompt import AnyPromptType

    from llmling_agent.agent import AnyAgent
    from llmling_agent.delegation.agentgroup import Team
    from llmling_agent.models.forward_targets import ConnectionType

TContent = TypeVar("TContent")
FilterFn = Callable[[ChatMessage[Any]], bool]
TransformFn = Callable[[ChatMessage[TContent]], ChatMessage[TContent]]
QueueStrategy = Literal["concat", "latest", "buffer"]
logger = get_logger(__name__)


@dataclass(frozen=True)
class TalkStats:
    """Statistics for a single connection."""

    source_name: str | None
    target_names: set[str]
    messages: list[ChatMessage[Any]] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)

    @property
    def message_count(self) -> int:
        """Number of messages transmitted."""
        return len(self.messages)

    @property
    def last_message_time(self) -> datetime | None:
        """When the last message was sent."""
        return self.messages[-1].timestamp if self.messages else None

    @property
    def token_count(self) -> int:
        """Total tokens used."""
        return sum(
            msg.cost_info.token_usage["total"] for msg in self.messages if msg.cost_info
        )

    @property
    def byte_count(self) -> int:
        """Total bytes transmitted."""
        return sum(len(str(msg.content).encode()) for msg in self.messages)

    def format(
        self,
        style: FormatStyle = "simple",
        **kwargs: Any,
    ) -> str:
        """Format the conversation that happened on this connection."""
        return "\n".join(msg.format(style, **kwargs) for msg in self.messages)


@dataclass
class TeamTalkStats:
    """Statistics aggregated from multiple connections."""

    stats: list[TalkStats | TeamTalkStats] = field(default_factory=list)

    # def __init__(self, stats: list[TalkStats | TeamTalkStats]):
    #     self.stats = stats

    @property
    def messages(self) -> list[ChatMessage[Any]]:
        """All messages across all connections, flattened."""
        return [msg for stat in self.stats for msg in stat.messages]

    @property
    def message_count(self) -> int:
        """Total messages across all connections."""
        return len(self.messages)

    @property
    def start_time(self) -> datetime:
        """Total messages across all connections."""
        return self.stats[0].start_time

    @property
    def num_connections(self) -> int:
        """Number of active connections."""
        return len(self.stats)

    @property
    def token_count(self) -> int:
        """Total tokens across all connections."""
        return sum(
            msg.cost_info.token_usage["total"] for msg in self.messages if msg.cost_info
        )

    @property
    def byte_count(self) -> int:
        """Total bytes transmitted."""
        return sum(len(str(msg.content).encode()) for msg in self.messages)

    @property
    def last_message_time(self) -> datetime | None:
        """Most recent message time."""
        if not self.messages:
            return None
        return max(msg.timestamp for msg in self.messages)

    def format(
        self,
        style: FormatStyle = "simple",
        **kwargs: Any,
    ) -> str:
        """Format all conversations in the team."""
        return "\n".join(msg.format(style, **kwargs) for msg in self.messages)

    @cached_property
    def source_names(self) -> set[str]:
        """Set of unique source names recursively."""

        def _collect_source_names(stat: TalkStats | TeamTalkStats) -> set[str]:
            """Recursively collect source names."""
            if isinstance(stat, TalkStats):
                return {stat.source_name} if stat.source_name else set()
            # It's a TeamTalkStats, recurse
            names = set()
            for s in stat.stats:
                names.update(_collect_source_names(s))
            return names

        names = set()
        for stat in self.stats:
            names.update(_collect_source_names(stat))
        return names

    @cached_property
    def target_names(self) -> set[str]:
        """Set of all target names across connections."""
        return {name for s in self.stats for name in s.target_names}


class Talk[TTransmittedData]:
    """Manages message flow between agents/groups."""

    message_received = Signal(ChatMessage[TTransmittedData])  # Original message
    message_forwarded = Signal(ChatMessage[Any])  # After any transformation

    def __init__(
        self,
        source: AnyAgent[Any, TTransmittedData],
        targets: list[AnyAgent[Any, Any]],
        group: TeamTalk | None = None,
        *,
        connection_type: ConnectionType = "run",
        wait_for_connections: bool = False,
        priority: int = 0,
        delay: timedelta | None = None,
        queued: bool = False,
        queue_strategy: QueueStrategy = "latest",
    ):
        """Initialize talk connection.

        Args:
            source: Agent sending messages
            targets: Agents receiving messages
            group: Optional group this talk belongs to
            connection_type: How to handle messages:
                - "run": Execute message as a new run in target
                - "context": Add message as context to target
                - "forward": Forward message to target's outbox
            wait_for_connections: Whether to wait for all targets to complete
            priority: Task priority (lower = higher priority)
            delay: Optional delay before processing
            queued: Whether messages should be queued for manual processing
            queue_strategy: How to process queued messages:
                - "concat": Combine all messages with newlines
                - "latest": Use only the most recent message
                - "buffer": Process all messages individually
        """
        self.source = source
        self.targets = targets
        self.group = group
        self.priority = priority
        self.delay = delay
        self.active = True
        self.connection_type = connection_type
        self.wait_for_connections = wait_for_connections
        self.queued = queued
        self.queue_strategy = queue_strategy
        self._pending_messages: list[ChatMessage[TTransmittedData]] = []
        names = {t.name for t in targets}
        self._stats = TalkStats(source_name=source.name, target_names=names)
        self._filter: FilterFn | None = None
        self._transformer: TransformFn | None = None

    async def _handle_message(
        self,
        message: ChatMessage[TTransmittedData],
        prompt: str | None = None,
    ):
        """Handle message forwarding based on connection configuration."""
        logger.debug(
            "Message from %s to %s: %r (type: %s) (prompt: %s)",
            self.source.name,
            [t.name for t in self.targets],
            message.content,
            self.connection_type,
            prompt,
        )
        self.source.outbox.emit(message, None)

        if not self.active or (self.group and not self.group.active):
            return
        if self._filter and not self._filter(message):
            return

        # Update stats
        self._stats = replace(
            self._stats,
            messages=[*self._stats.messages, message],
        )

        # For queued connections, just queue the message
        if self.queued:
            self._pending_messages.append(message)
            return

        # Process message immediately
        await self._process_message(message, prompt)

    async def _process_message(
        self,
        message: ChatMessage[TTransmittedData],
        prompt: str | None = None,
    ) -> list[ChatMessage[Any]]:
        """Process a single message according to connection type.

        Returns:
            List of response messages (for "run" connections)
        """
        results: list[ChatMessage[Any]] = []

        match self.connection_type:
            case "run":
                for target in self.targets:
                    prompts: list[AnyPromptType] = [message.content]
                    if prompt:
                        prompts.append(prompt)
                    response = await target.run(*prompts)
                    response.forwarded_from.append(target.name)
                    target.outbox.emit(response, None)
                    results.append(response)

            case "context":
                for target in self.targets:

                    async def add_context(target=target):
                        meta = {
                            "type": "forwarded_message",
                            "role": message.role,
                            "model": message.model,
                            "cost_info": message.cost_info,
                            "timestamp": message.timestamp.isoformat(),
                            "prompt": prompt,
                        }
                        target.conversation.add_context_message(
                            str(message.content),
                            source=self.source.name,
                            metadata=meta,
                        )

                    if self.delay is not None or self.priority != 0:
                        target.run_background(
                            add_context(),
                            priority=self.priority,
                            delay=self.delay,
                        )
                    else:
                        target.run_task_sync(add_context())

            case "forward":
                for target in self.targets:
                    if self.delay is not None or self.priority != 0:

                        async def delayed_emit(target=target):
                            target.outbox.emit(message, prompt)

                        target.run_background(
                            delayed_emit(),
                            priority=self.priority,
                            delay=self.delay,
                        )
                    else:
                        target.outbox.emit(message, prompt)

        self.message_forwarded.emit(message)
        return results

    async def trigger(self) -> list[ChatMessage[TTransmittedData]]:
        """Process queued messages.

        Returns:
            List of processed messages:
            - concat/latest: Single merged message in list
            - buffer: All messages processed individually
        """
        if not self._pending_messages:
            return []

        match self.queue_strategy:
            case "buffer":
                # Process each message individually
                results: list[ChatMessage[TTransmittedData]] = []
                for message in self._pending_messages:
                    processed = await self._process_message(message, None)
                    results.append(message)
                    results.extend(processed)  # Add any responses
                self._pending_messages.clear()
                return results

            case "latest":
                # Just use the most recent message as-is
                merged = self._pending_messages[-1]
                responses = await self._process_message(merged, None)
                self._pending_messages.clear()
                return [merged, *responses]

            case "concat":
                # Ensure all messages have string content
                base = self._pending_messages[-1]
                contents = [str(m.content) for m in self._pending_messages]
                # Create merged message
                merged = replace(
                    base,
                    content="\n\n".join(contents),  # type: ignore
                    metadata={
                        **base.metadata,
                        "merged_count": len(self._pending_messages),
                        "queue_strategy": self.queue_strategy,
                    },
                )

                # Process the merged message
                responses = await self._process_message(merged, None)
                self._pending_messages.clear()
                return [merged, *responses]

        return []

    def when(self, condition: FilterFn) -> Self:
        """Add condition for message forwarding."""
        self._filter = condition
        return self

    @asynccontextmanager
    async def paused(self):
        """Temporarily set inactive."""
        previous = self.active
        self.active = False
        try:
            yield self
        finally:
            self.active = previous

    def disconnect(self):
        """Permanently disconnect the connection."""
        self.active = False

    @property
    def stats(self) -> TalkStats:
        """Get current connection statistics."""
        return self._stats


class TeamTalk(list["Talk | TeamTalk"]):
    """Group of connections with aggregate operations."""

    def __init__(self, talks: Sequence[Talk | TeamTalk]):
        super().__init__(talks)
        self._filter: FilterFn | None = None
        self.active = True

    @property
    def targets(self) -> list[AnyAgent[Any, Any]]:
        """Get all targets from all connections."""
        return [t for talk in self for t in talk.targets]

    async def _handle_message(self, message: ChatMessage[Any], prompt: str | None = None):
        for talk in self:
            await talk._handle_message(message, prompt)

    @classmethod
    def from_agents(
        cls,
        agents: Sequence[AnyAgent[Any, Any]],
        targets: list[AnyAgent[Any, Any]] | None = None,
    ) -> TeamTalk:
        """Create TeamTalk from a collection of agents."""
        return cls([Talk(agent, targets or []) for agent in agents])

    @asynccontextmanager
    async def paused(self):
        """Temporarily set inactive."""
        previous = self.active
        self.active = False
        try:
            yield self
        finally:
            self.active = previous

    def has_active_talks(self) -> bool:
        """Check if any contained talks are active."""
        return any(talk.active for talk in self)

    def get_active_talks(self) -> list[Talk | TeamTalk]:
        """Get list of currently active talks."""
        return [talk for talk in self if talk.active]

    @property
    def stats(self) -> TeamTalkStats:
        """Get aggregated statistics for all connections."""
        return TeamTalkStats(stats=[talk.stats for talk in self])

    def when(self, condition: FilterFn) -> Self:
        """Add condition to all connections in group."""
        for talk in self:
            talk.when(condition)
        return self

    def disconnect(self):
        """Disconnect all connections in group."""
        for talk in self:
            talk.disconnect()


class TalkManager:
    """Manages connections for both Agents and Teams."""

    agent_connected = Signal(object)  # Agent
    connection_added = Signal(Talk)  # Agent

    def __init__(self, owner: AnyAgent[Any, Any] | Team[Any]):
        self.owner = owner
        self._connections: list[Talk | TeamTalk] = []
        self._wait_states: dict[str, bool] = {}

    def set_wait_state(self, target: AnyAgent[Any, Any] | str, wait: bool = True):
        """Set waiting behavior for target."""
        target_name = target if isinstance(target, str) else target.name
        self._wait_states[target_name] = wait

    async def wait_for_connections(self, _seen: set[str] | None = None):
        """Wait for this agent and all connected agents to complete their tasks."""
        seen: set[str] = _seen or {self.owner.name}  # type: ignore

        # Wait for our own tasks
        await self.owner.complete_tasks()

        # Wait for connected agents
        for agent in self.get_targets():
            if agent.name not in seen:
                seen.add(agent.name)
                await agent.connections.wait_for_connections(seen)

    def get_targets(
        self, recursive: bool = False, _seen: set[str] | None = None
    ) -> set[AnyAgent[Any, Any]]:
        """Get all currently connected target agents.

        Args:
            recursive: Whether to include targets of targets
        """
        # Get direct targets
        targets = {t for conn in self._connections for t in conn.targets if conn.active}

        if not recursive:
            return targets

        # Track seen agents to prevent cycles
        seen = _seen or {self.owner.name}  # type: ignore
        all_targets = set()

        for target in targets:
            if target.name not in seen:
                _targets = target.connections.get_targets(recursive=True, _seen=seen)
                seen.add(target.name)
                all_targets.add(target)
                all_targets.update(_targets)

        return all_targets

    def has_connection_to(self, target: AnyAgent[Any, Any]) -> bool:
        """Check if target is connected."""
        return any(target in conn.targets for conn in self._connections if conn.active)

    @overload
    def connect_agent_to(
        self,
        other: AnyAgent[Any, Any] | str,
        *,
        connection_type: ConnectionType = "run",
        priority: int = 0,
        delay: timedelta | None = None,
        queued: bool = False,
        queue_strategy: Literal["concat", "latest", "buffer"] = "latest",
    ) -> Talk[Any]: ...

    @overload
    def connect_agent_to(
        self,
        other: Team[Any],
        *,
        connection_type: ConnectionType = "run",
        priority: int = 0,
        delay: timedelta | None = None,
        queued: bool = False,
        queue_strategy: Literal["concat", "latest", "buffer"] = "latest",
    ) -> TeamTalk: ...

    def connect_agent_to(
        self,
        other: AnyAgent[Any, Any] | Team[Any] | str,
        *,
        connection_type: ConnectionType = "run",
        priority: int = 0,
        delay: timedelta | None = None,
        queued: bool = False,
        queue_strategy: Literal["concat", "latest", "buffer"] = "latest",
    ) -> Talk[Any] | TeamTalk:
        """Handle single agent connections."""
        from llmling_agent.agent import Agent, StructuredAgent
        from llmling_agent.delegation.agentgroup import Team

        if not isinstance(self.owner, Agent | StructuredAgent):
            msg = "connect_agent_to can only be used with single agents"
            raise TypeError(msg)

        targets = self._resolve_targets(other)
        for target in targets:
            self.agent_connected.emit(target)

        if isinstance(other, Team):
            conns = [
                self.add_connection(
                    self.owner,
                    [target],
                    connection_type=connection_type,
                    priority=priority,
                    delay=delay,
                    queued=queued,
                    queue_strategy=queue_strategy,
                )
                for target in targets
            ]
            return TeamTalk(conns)

        return self.add_connection(
            self.owner,
            targets,
            connection_type=connection_type,
            priority=priority,
            delay=delay,
            queued=queued,
            queue_strategy=queue_strategy,
        )

    def add_connection(
        self,
        source: AnyAgent[Any, Any],
        targets: list[AnyAgent[Any, Any]],
        *,
        connection_type: ConnectionType = "run",
        priority: int = 0,
        delay: timedelta | None = None,
        queued: bool = False,
        queue_strategy: Literal["concat", "latest", "buffer"] = "latest",
    ) -> Talk[Any]:
        """Add a connection to the manager."""
        connection = Talk(
            source,
            targets,
            connection_type=connection_type,
            priority=priority,
            delay=delay,
            queued=queued,
            queue_strategy=queue_strategy,
        )
        self.connection_added.emit(connection)
        self._connections.append(connection)
        return connection

    def connect_group_to(
        self,
        other: AnyAgent[Any, Any] | Team[Any] | str,
        *,
        connection_type: ConnectionType = "run",
        priority: int = 0,
        delay: timedelta | None = None,
        queued: bool = False,
        queue_strategy: Literal["concat", "latest", "buffer"] = "latest",
        **kwargs: Any,
    ) -> TeamTalk:
        """Handle group connections."""
        from llmling_agent.delegation.agentgroup import Team

        if not isinstance(self.owner, Team):
            msg = "connect_group_to can only be used with agent groups"
            raise TypeError(msg)

        targets = self._resolve_targets(other)
        for target in targets:
            self.agent_connected.emit(target)

        conns = [
            self.add_connection(
                src,
                [t],
                connection_type=connection_type,
                priority=priority,
                delay=delay,
                queued=queued,
                queue_strategy=queue_strategy,
            )
            for src in self.owner.agents
            for t in targets
        ]
        return TeamTalk(conns)

    def _resolve_targets(
        self, other: AnyAgent[Any, Any] | Team[Any] | str
    ) -> list[AnyAgent[Any, Any]]:
        """Resolve target(s) to connect to."""
        from llmling_agent.agent import Agent, StructuredAgent
        from llmling_agent.delegation.agentgroup import Team

        match other:
            case str():
                if (
                    not isinstance(self.owner, Agent | StructuredAgent)
                    or not self.owner.context.pool
                ):
                    msg = "Pool required for forwarding to agent by name"
                    raise ValueError(msg)
                return [self.owner.context.pool.get_agent(other)]
            case Team():
                return list(other.agents)
        return [other]

    async def trigger_all(self) -> dict[str, list[ChatMessage[Any]]]:
        """Trigger all queued connections."""
        results = {}
        for talk in self._connections:
            if isinstance(talk, Talk) and talk.queued:
                results[talk.source.name] = await talk.trigger()
        return results

    async def trigger_for(
        self, target: str | AnyAgent[Any, Any]
    ) -> list[ChatMessage[Any]]:
        """Trigger queued connections to specific target."""
        target_name = target if isinstance(target, str) else target.name
        results = []
        for talk in self._connections:
            if isinstance(talk, Talk) and talk.queued:  # noqa: SIM102
                if any(t.name == target_name for t in talk.targets):
                    results.extend(await talk.trigger())
        return results

    def disconnect_all(self):
        """Disconnect all managed connections."""
        for conn in self._connections:
            conn.disconnect()
        self._connections.clear()

    def disconnect(self, agent: AnyAgent[Any, Any]):
        """Disconnect a specific agent."""
        to_disconnect: list[Talk | TeamTalk] = []
        for talk in self._connections:
            match talk:
                case Talk():
                    if agent in talk.targets or agent == talk.source:
                        to_disconnect.append(talk)
                case TeamTalk():
                    if agent in talk.targets:
                        to_disconnect.append(talk)

        for talk in to_disconnect:
            talk.active = False
            self._connections.remove(talk)

    async def route_message(self, message: ChatMessage[Any], wait: bool | None = None):
        """Route message to all connections."""
        if wait is not None:
            should_wait = wait
        else:
            should_wait = any(
                self._wait_states.get(t.name, False) for t in self.get_targets()
            )

        logger.debug(
            "TalkManager routing message from %s to %d connections",
            message.name,
            len(self._connections),
        )

        forwarded_from = [*message.forwarded_from, self.owner.name]  # type: ignore
        message_copy = replace(message, forwarded_from=forwarded_from)

        for talk in self._connections:
            await talk._handle_message(message_copy, None)

        if should_wait:
            await self.wait_for_connections()

    @asynccontextmanager
    async def paused_routing(self):
        """Temporarily pause message routing to connections."""
        active_talks = [talk for talk in self._connections if talk.active]
        for talk in active_talks:
            talk.active = False

        try:
            yield self
        finally:
            for talk in active_talks:
                talk.active = True

    @property
    def stats(self) -> TeamTalkStats:
        """Get aggregated statistics for all connections."""
        return TeamTalkStats(stats=[conn.stats for conn in self._connections])

    def get_connections(self, recursive: bool = False) -> list[Talk[Any]]:
        """Get all Talk connections, flattening TeamTalks."""

        def _collect_talks(
            item: Talk[Any] | TeamTalk, seen: set[str] | None = None
        ) -> list[Talk[Any]]:
            match item:
                case Talk():
                    return [item]
                case TeamTalk():
                    if not recursive:
                        return [t for subitem in item for t in _collect_talks(subitem)]

                    seen = seen or {self.owner.name}  # type: ignore
                    talks = []

                    for subitem in item:
                        talks.extend(_collect_talks(subitem))

                    for target in item.targets:
                        if target.name not in seen:
                            seen.add(target.name)
                            conns = target.connections.get_connections(recursive=True)
                            talks.extend(conns)
                    return talks

        return [talk for conn in self._connections for talk in _collect_talks(conn)]
