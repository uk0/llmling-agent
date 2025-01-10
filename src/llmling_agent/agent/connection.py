"""Manages message flow between agents/groups."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from functools import cached_property
from typing import TYPE_CHECKING, Any, Self, overload

from psygnal import Signal
from typing_extensions import TypeVar

from llmling_agent.models.messages import ChatMessage


if TYPE_CHECKING:
    from llmling_agent.agent import AnyAgent
    from llmling_agent.delegation.agentgroup import Team

TContent = TypeVar("TContent")
FilterFn = Callable[[ChatMessage[Any]], bool]
TransformFn = Callable[[ChatMessage[TContent]], ChatMessage[TContent]]


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


class Talk:
    """Manages message flow between agents/groups."""

    message_received = Signal(ChatMessage[Any])  # Original message
    message_forwarded = Signal(ChatMessage[Any])  # After any transformation

    def __init__(
        self,
        source: AnyAgent[Any, Any],
        targets: list[AnyAgent[Any, Any]],
        group: TeamTalk | None = None,
    ):
        self.source = source
        self.targets = targets
        self.group = group
        self.active = True
        self._stats = TalkStats(
            source_name=source.name, target_names={t.name for t in targets}
        )
        self._filter: FilterFn | None = None
        self._transformer: TransformFn | None = None
        source.outbox.connect(self._handle_message)

    def _handle_message(self, message: ChatMessage[Any], prompt: str | None = None):
        # We always receive a message
        self.message_received.emit(message)
        if not self.active or (self.group and not self.group.active):
            return
        if self._filter and not self._filter(message):
            return
        # Only update stats and forward if connection is active
        self._stats = TalkStats(
            message_count=self._stats.message_count + 1,
            start_time=self._stats.start_time,
            last_message_time=datetime.now(),
            token_count=self._stats.token_count
            + (message.cost_info.token_usage["total"] if message.cost_info else 0),
            byte_count=self._stats.byte_count + len(str(message.content).encode()),
            source_name=self._stats.source_name,
            target_names=self._stats.target_names,
        )

        # for target in self.targets:
        #     target._handle_message(message, prompt)
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
        self.source.outbox.disconnect(self._handle_message)
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

    def _handle_message(self, message: ChatMessage[Any], prompt: str | None = None):
        for talk in self:
            talk._handle_message(message, prompt)

    @classmethod
    def from_agents(
        cls,
        agents: list[AnyAgent[Any, Any]],
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

    def __init__(self, owner: AnyAgent[Any, Any] | Team[Any]):
        self.owner = owner
        self._connections: list[Talk | TeamTalk] = []

    def _route_message(self, message: ChatMessage[Any], prompt: str | None):
        # Each Talk already knows its targets
        for talk in self._connections:
            talk._handle_message(message, prompt)

    def get_targets(self) -> set[AnyAgent[Any, Any]]:
        """Get all currently connected target agents."""
        return {t for conn in self._connections for t in conn.targets if conn.active}

    def has_connection_to(self, target: AnyAgent[Any, Any]) -> bool:
        """Check if target is connected."""
        return any(target in conn.targets for conn in self._connections if conn.active)

    @overload
    def connect_agent_to(
        self,
        other: AnyAgent[Any, Any] | str,
        **kwargs: Any,
    ) -> Talk: ...

    @overload
    def connect_agent_to(
        self,
        other: Team[Any],
        **kwargs: Any,
    ) -> TeamTalk: ...

    def connect_agent_to(
        self,
        other: AnyAgent[Any, Any] | Team[Any] | str,
        **kwargs: Any,
    ) -> Talk | TeamTalk:
        """Handle single agent connections."""
        from llmling_agent.agent import Agent, StructuredAgent
        from llmling_agent.delegation.agentgroup import Team

        if not isinstance(self.owner, Agent | StructuredAgent):
            msg = "connect_agent_to can only be used with single agents"
            raise TypeError(msg)

        targets = self._resolve_targets(other)
        if isinstance(other, Team):
            conns = [Talk(self.owner, [target]) for target in targets]
            connections = TeamTalk(conns)
            self._connections.extend(connections)
            return connections

        connection = Talk(self.owner, targets)
        self._connections.append(connection)
        return connection

    def connect_group_to(
        self,
        other: AnyAgent[Any, Any] | Team[Any] | str,
        **kwargs: Any,
    ) -> TeamTalk:
        """Handle group connections."""
        from llmling_agent.delegation.agentgroup import Team

        if not isinstance(self.owner, Team):
            msg = "connect_group_to can only be used with agent groups"
            raise TypeError(msg)

        targets = self._resolve_targets(other)
        conns = [Talk(src, [t]) for src in self.owner.agents for t in targets]
        connections = TeamTalk(conns)
        self._connections.extend(connections)
        return connections

    def _resolve_targets(
        self, other: AnyAgent[Any, Any] | Team[Any] | str
    ) -> list[AnyAgent[Any, Any]]:
        """Resolve target(s) to connect to."""
        from llmling_agent.agent import Agent, StructuredAgent
        from llmling_agent.delegation.agentgroup import Team

        if isinstance(other, str):
            if (
                not isinstance(self.owner, Agent | StructuredAgent)
                or not self.owner.context.pool
            ):
                msg = "Pool required for forwarding to agent by name"
                raise ValueError(msg)
            return [self.owner.context.pool.get_agent(other)]
        if isinstance(other, Team):
            return other.agents
        return [other]

    def disconnect_all(self) -> None:
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
