"""Manages message flow between agents/groups."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from functools import cached_property
from typing import TYPE_CHECKING, Any, Self, overload

from psygnal import Signal
from typing_extensions import TypeVar

from llmling_agent.log import get_logger
from llmling_agent.models.messages import ChatMessage


if TYPE_CHECKING:
    from toprompt import AnyPromptType

    from llmling_agent.agent import AnyAgent
    from llmling_agent.delegation.agentgroup import Team
    from llmling_agent.models.forward_targets import ConnectionType

TContent = TypeVar("TContent")
FilterFn = Callable[[ChatMessage[Any]], bool]
TransformFn = Callable[[ChatMessage[TContent]], ChatMessage[TContent]]

logger = get_logger(__name__)


@dataclass(frozen=True)
class TalkStats:
    """Statistics for a single connection."""

    message_count: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    last_message_time: datetime | None = None
    token_count: int = 0
    byte_count: int = 0
    source_name: str | None = None
    target_names: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class TeamTalkStats:
    """Statistics aggregated from multiple connections."""

    stats: list[TalkStats | TeamTalkStats] = field(default_factory=list)

    @cached_property
    def message_count(self) -> int:
        """Total messages across all connections."""
        return sum(stat.message_count for stat in self.stats)

    @cached_property
    def token_count(self) -> int:
        """Total tokens across all connections."""
        return sum(stat.token_count for stat in self.stats)

    @cached_property
    def byte_count(self) -> int:
        """Total bytes forwarded across all connections."""
        return sum(stat.byte_count for stat in self.stats)

    @property
    def num_connections(self) -> int:
        """Number of active connections."""
        return len(self.stats)

    @cached_property
    def start_time(self) -> datetime:
        """Earliest connection start time."""
        if not self.stats:
            return datetime.now()
        return min(stat.start_time for stat in self.stats)

    @cached_property
    def last_message_time(self) -> datetime | None:
        """Most recent message across all connections."""
        times = [s.last_message_time for s in self.stats if s.last_message_time]
        return max(times) if times else None

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
        """
        self.source = source
        self.targets = targets
        self.group = group
        self.priority = priority
        self.delay = delay
        self.active = True
        self.connection_type = connection_type
        self.wait_for_connections = wait_for_connections
        names = {t.name for t in targets}
        self._stats = TalkStats(source_name=source.name, target_names=names)
        self._filter: FilterFn | None = None
        self._transformer: TransformFn | None = None

    async def _handle_message(
        self,
        message: ChatMessage[TTransmittedData],
        prompt: str | None = None,
    ):
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
        totals = message.cost_info.token_usage["total"] if message.cost_info else 0
        self._stats = TalkStats(
            message_count=self._stats.message_count + 1,
            start_time=self._stats.start_time,
            last_message_time=datetime.now(),
            token_count=self._stats.token_count + totals,
            byte_count=self._stats.byte_count + len(str(message.content).encode()),
            source_name=self._stats.source_name,
            target_names=self._stats.target_names,
        )

        # Handle message based on connection type
        match self.connection_type:
            case "run":
                for target in self.targets:
                    prompts: list[AnyPromptType] = [message.content]
                    if prompt:
                        prompts.append(prompt)
                    response = await target.run(*prompts)
                    response.forwarded_from.append(target.name)
                    target.outbox.emit(response, None)

            case "context":
                for target in self.targets:

                    async def add_context(target=target):
                        target.conversation.add_context_message(
                            str(message.content),
                            source=self.source.name,
                            metadata={
                                "type": "forwarded_message",
                                "role": message.role,
                                "model": message.model,
                                "cost_info": message.cost_info,
                                "timestamp": message.timestamp.isoformat(),
                                "prompt": prompt,
                            },
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
        # Track seen agents to avoid cycles
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
                # Get recursive targets
                all_targets.update(_targets)

        return all_targets

    def has_connection_to(self, target: AnyAgent[Any, Any]) -> bool:
        """Check if target is connected."""
        return any(target in conn.targets for conn in self._connections if conn.active)

    @overload
    def connect_agent_to(
        self,
        other: AnyAgent[Any, Any] | str,
        connection_type: ConnectionType = "run",
        priority: int = 0,
        delay: timedelta | None = None,
    ) -> Talk: ...

    @overload
    def connect_agent_to(
        self,
        other: Team[Any],
        connection_type: ConnectionType = "run",
        priority: int = 0,
        delay: timedelta | None = None,
    ) -> TeamTalk: ...

    def connect_agent_to(
        self,
        other: AnyAgent[Any, Any] | Team[Any] | str,
        connection_type: ConnectionType = "run",
        priority: int = 0,
        delay: timedelta | None = None,
    ) -> Talk | TeamTalk:
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
        )

    def add_connection(
        self,
        source: AnyAgent[Any, Any],
        targets: list[AnyAgent[Any, Any]],
        connection_type: ConnectionType = "run",
        priority: int = 0,
        delay: timedelta | None = None,
    ) -> Talk:
        """Add a connection to the manager."""
        connection = Talk(
            source,
            targets,
            connection_type=connection_type,
            priority=priority,
            delay=delay,
        )
        self.connection_added.emit(connection)
        self._connections.append(connection)
        return connection

    def connect_group_to(
        self,
        other: AnyAgent[Any, Any] | Team[Any] | str,
        connection_type: ConnectionType = "run",
        priority: int = 0,
        delay: timedelta | None = None,
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

    @property
    def stats(self) -> TeamTalkStats:
        """Get aggregated statistics for all connections."""
        return TeamTalkStats(stats=[conn.stats for conn in self._connections])

    def get_connections(self, recursive: bool = False) -> list[Talk]:
        """Get all Talk connections, flattening TeamTalks.

        Args:
            recursive: Whether to include connections from nested TeamTalks.
                    If True, follows TeamTalks through their targets' connections.

        Returns:
            List of all individual Talk connections
        """

        def _collect_talks(
            item: Talk | TeamTalk, seen: set[str] | None = None
        ) -> list[Talk]:
            match item:
                case Talk():
                    return [item]
                case TeamTalk():
                    if not recursive:
                        return [
                            talk for subitem in item for talk in _collect_talks(subitem)
                        ]

                    # Handle recursive case
                    seen = seen or {self.owner.name}  # type: ignore[has-type]
                    talks = []

                    # First get direct talks from this TeamTalk
                    for subitem in item:
                        talks.extend(_collect_talks(subitem))

                    # Then recursively get talks from targets if not seen
                    for target in item.targets:
                        if target.name not in seen:
                            seen.add(target.name)
                            talks.extend(
                                target.connections.get_connections(recursive=True)
                            )

                    return talks

        return [talk for conn in self._connections for talk in _collect_talks(conn)]

    async def route_message(self, message: ChatMessage[Any], wait: bool | None = None):
        """Route message to all connections.

        Args:
            message: Message to route
            wait: Override default waiting behavior
                 None = use configured states from _wait_states
                 True/False = override for this message
        """
        if wait is not None:
            should_wait = wait
        else:
            # Use configured states as fallback
            should_wait = any(
                self._wait_states.get(t.name, False) for t in self.get_targets()
            )
        msg = "TalkManager routing message from %s to %d connections"
        logger.debug(msg, message.content, len(self._connections))
        forwarded_from = [*message.forwarded_from, self.owner.name]  # type: ignore[has-type]
        message_copy = replace(message, forwarded_from=forwarded_from)
        for talk in self._connections:
            await talk._handle_message(message_copy, None)
        if should_wait:
            await self.wait_for_connections()
