"""Manages message flow between agents/groups."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import replace
from typing import TYPE_CHECKING, Any, overload

from psygnal import Signal

from llmling_agent.log import get_logger
from llmling_agent.talk import Talk, TeamTalk, TeamTalkStats


if TYPE_CHECKING:
    from datetime import timedelta

    from llmling_agent.agent import AnyAgent
    from llmling_agent.common_types import AgentName, AnyFilterFn, AnyTransformFn
    from llmling_agent.delegation.team import Team
    from llmling_agent.models.forward_targets import ConnectionType
    from llmling_agent.models.messages import ChatMessage
    from llmling_agent.talk.talk import QueueStrategy

logger = get_logger(__name__)


class ConnectionManager:
    """Manages connections for both Agents and Teams."""

    agent_connected = Signal(object)  # Agent
    connection_added = Signal(Talk)  # Agent

    def __init__(self, owner: AnyAgent[Any, Any] | Team[Any]):
        self.owner = owner
        self._connections: list[Talk | TeamTalk] = []
        self._wait_states: dict[AgentName, bool] = {}

    def __repr__(self):
        return f"ConnectionManager({self.owner})"

    def set_wait_state(self, target: AnyAgent[Any, Any] | AgentName, wait: bool = True):
        """Set waiting behavior for target."""
        target_name = target if isinstance(target, str) else target.name
        self._wait_states[target_name] = wait

    async def wait_for_connections(self, _seen: set[AgentName] | None = None):
        """Wait for this agent and all connected agents to complete their tasks."""
        seen: set[AgentName] = _seen or {self.owner.name}  # type: ignore

        # Wait for our own tasks
        await self.owner.complete_tasks()

        # Wait for connected agents
        for agent in self.get_targets():
            if agent.name not in seen:
                seen.add(agent.name)
                await agent.connections.wait_for_connections(seen)

    def get_targets(
        self, recursive: bool = False, _seen: set[AgentName] | None = None
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
        other: AnyAgent[Any, Any] | AgentName,
        *,
        connection_type: ConnectionType = "run",
        priority: int = 0,
        delay: timedelta | None = None,
        queued: bool = False,
        queue_strategy: QueueStrategy = "latest",
        transform: AnyTransformFn | None = None,
        filter_condition: AnyFilterFn | None = None,
        stop_condition: AnyFilterFn | None = None,
        exit_condition: AnyFilterFn | None = None,
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
        queue_strategy: QueueStrategy = "latest",
        transform: AnyTransformFn | None = None,
        filter_condition: AnyFilterFn | None = None,
        stop_condition: AnyFilterFn | None = None,
        exit_condition: AnyFilterFn | None = None,
    ) -> TeamTalk: ...

    def connect_agent_to(
        self,
        other: AnyAgent[Any, Any] | Team[Any] | AgentName,
        *,
        connection_type: ConnectionType = "run",
        priority: int = 0,
        delay: timedelta | None = None,
        queued: bool = False,
        queue_strategy: QueueStrategy = "latest",
        transform: AnyTransformFn | None = None,
        filter_condition: AnyFilterFn | None = None,
        stop_condition: AnyFilterFn | None = None,
        exit_condition: AnyFilterFn | None = None,
    ) -> Talk[Any] | TeamTalk:
        """Handle single agent connections."""
        from llmling_agent.agent import Agent, StructuredAgent
        from llmling_agent.delegation.team import Team

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
                    transform=transform,
                    filter_condition=filter_condition,
                    stop_condition=stop_condition,
                    exit_condition=exit_condition,
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
            transform=transform,
            filter_condition=filter_condition,
            stop_condition=stop_condition,
            exit_condition=exit_condition,
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
        queue_strategy: QueueStrategy = "latest",
        transform: AnyTransformFn | None = None,
        filter_condition: AnyFilterFn | None = None,
        stop_condition: AnyFilterFn | None = None,
        exit_condition: AnyFilterFn | None = None,
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
            transform=transform,
            filter_condition=filter_condition,
            stop_condition=stop_condition,
            exit_condition=exit_condition,
        )
        self.connection_added.emit(connection)
        self._connections.append(connection)
        return connection

    def connect_group_to(
        self,
        other: AnyAgent[Any, Any] | Team[Any] | AgentName,
        *,
        connection_type: ConnectionType = "run",
        priority: int = 0,
        delay: timedelta | None = None,
        queued: bool = False,
        queue_strategy: QueueStrategy = "latest",
        transform: AnyTransformFn | None = None,
        filter_condition: AnyFilterFn | None = None,
        stop_condition: AnyFilterFn | None = None,
        exit_condition: AnyFilterFn | None = None,
        **kwargs: Any,
    ) -> TeamTalk:
        """Handle group connections."""
        from llmling_agent.delegation.team import Team

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
                transform=transform,
                filter_condition=filter_condition,
                stop_condition=stop_condition,
                exit_condition=exit_condition,
            )
            for src in self.owner.agents
            for t in targets
        ]
        return TeamTalk(conns)

    def _resolve_targets(
        self, other: AnyAgent[Any, Any] | Team[Any] | AgentName
    ) -> list[AnyAgent[Any, Any]]:
        """Resolve target(s) to connect to."""
        from llmling_agent.agent import Agent, StructuredAgent
        from llmling_agent.delegation.team import Team

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

    async def trigger_all(self) -> dict[AgentName, list[ChatMessage[Any]]]:
        """Trigger all queued connections."""
        results = {}
        for talk in self._connections:
            if isinstance(talk, Talk) and talk.queued:
                results[talk.source.name] = await talk.trigger()
        return results

    async def trigger_for(
        self, target: AgentName | AnyAgent[Any, Any]
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
        msg = "ConnectionManager routing message from %s to %d connections"
        logger.debug(msg, message.name, len(self._connections))

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
        result = []
        seen = set()

        # Get our direct connections
        for conn in self._connections:
            if isinstance(conn, Talk):
                result.append(conn)
            else:  # TeamTalk
                result.extend(t for t in conn if isinstance(t, Talk))

        # Get target connections if recursive
        if recursive:
            for conn in result:
                for target in conn.targets:
                    if target.name not in seen:
                        seen.add(target.name)
                        result.extend(target.connections.get_connections(True))

        return result

    def get_mermaid_diagram(
        self,
        include_details: bool = True,
        recursive: bool = True,
    ) -> str:
        """Generate mermaid flowchart of all connections."""
        lines = ["flowchart LR"]
        connections = self.get_connections(recursive=recursive)

        for talk in connections:
            source = talk.source.name
            for target in talk.targets:
                if include_details:
                    details: list[str] = []
                    details.append(talk.connection_type)
                    if talk.queued:
                        details.append(f"queued({talk.queue_strategy})")
                    if talk._filter_condition:
                        details.append(f"filter:{talk._filter_condition.__name__}")
                    if talk._stop_condition:
                        details.append(f"stop:{talk._stop_condition.__name__}")
                    if talk._exit_condition:
                        details.append(f"exit:{talk._exit_condition.__name__}")
                    elif any([
                        talk._filter_condition,
                        talk._stop_condition,
                        talk._exit_condition,
                    ]):
                        details.append("conditions")

                    label = f"|{' '.join(details)}|" if details else ""
                    lines.append(f"    {source}--{label}-->{target.name}")
                else:
                    lines.append(f"    {source}-->{target.name}")

        return "\n".join(lines)


if __name__ == "__main__":
    from llmling_agent.agent import Agent

    agent = Agent[None]("test_agent")
    agent_2 = Agent[None]("test_agent_2")
    agent_3 = Agent[None]("test_agent_3")
    agent_4 = Agent[None]("test_agent_3")
    _conn_1 = agent >> agent_2
    _conn_2 = agent >> agent_3
    _conn_3 = agent_2 >> agent_4
    print(agent.connections.get_connections(recursive=True))
    print(agent.connections.get_mermaid_diagram())
