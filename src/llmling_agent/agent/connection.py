"""Manages message flow between agents/groups."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import replace
import inspect
from typing import TYPE_CHECKING, Any, Literal, Self, overload

from psygnal import Signal
from typing_extensions import TypeVar

from llmling_agent.log import get_logger
from llmling_agent.models.messages import ChatMessage
from llmling_agent.talk.stats import TalkStats, TeamTalkStats


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence
    from datetime import timedelta

    from toprompt import AnyPromptType

    from llmling_agent.agent import AnyAgent
    from llmling_agent.common_types import AnyFilterFn
    from llmling_agent.delegation.agentgroup import Team
    from llmling_agent.models.forward_targets import ConnectionType

TContent = TypeVar("TContent")
QueueStrategy = Literal["concat", "latest", "buffer"]
logger = get_logger(__name__)


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
        transform: Callable[[Any], Any | Awaitable[Any]] | None = None,
        filter_condition: AnyFilterFn | None = None,
        stop_condition: AnyFilterFn | None = None,
        exit_condition: AnyFilterFn | None = None,
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
            transform: Optional function to transform messages
            filter_condition: Optional condition for filtering messages
            stop_condition: Optional condition for disconnecting
            exit_condition: Optional condition for stopping the event loop
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
        self._transform = transform
        self._filter_condition = filter_condition
        self._stop_condition = stop_condition
        self._exit_condition = exit_condition

    def __repr__(self):
        targets = [t.name for t in self.targets]
        return f"<Talk({self.connection_type}) {self.source.name} -> {targets}>"

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
        # Check active state
        if not self.active or (self.group and not self.group.active):
            return

        # Check exit condition
        if self._exit_condition:
            do_exit = self._exit_condition(message, self.stats)
            if inspect.isawaitable(do_exit):
                if await do_exit:
                    raise SystemExit
            elif do_exit:
                raise SystemExit

        # Check stop condition
        if self._stop_condition:
            do_stop = self._stop_condition(message, self.stats)
            if inspect.isawaitable(do_stop):
                if await do_stop:
                    self.disconnect()
                    return
            elif do_stop:
                self.disconnect()
                return
        # Check filter condition
        if self._filter_condition:
            do_filter = self._filter_condition(message, self.stats)
            if inspect.isawaitable(do_filter):
                if not await do_filter:
                    return
            elif not do_filter:
                return
        # Apply transform if configured
        if self._transform:
            transformed = self._transform(message)
            if inspect.isawaitable(transformed):
                message = await transformed
            else:
                message = transformed
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
                meta = {
                    "type": "forwarded_message",
                    "role": message.role,
                    "model": message.model,
                    "cost_info": message.cost_info,
                    "timestamp": message.timestamp.isoformat(),
                    "prompt": prompt,
                }
                for target in self.targets:

                    async def add_context(target=target):
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
                latest = self._pending_messages[-1]
                responses = await self._process_message(latest, None)
                self._pending_messages.clear()
                return [latest, *responses]

            case "concat":
                # Ensure all messages have string content
                base = self._pending_messages[-1]
                contents = [str(m.content) for m in self._pending_messages]
                # Create merged message
                meta = {
                    **base.metadata,
                    "merged_count": len(self._pending_messages),
                    "queue_strategy": self.queue_strategy,
                }
                content = "\n\n".join(contents)
                merged = replace(base, content=content, metadata=meta)  # type: ignore

                # Process the merged message
                responses = await self._process_message(merged, None)
                self._pending_messages.clear()
                return [merged, *responses]

        return []

    def when(self, condition: AnyFilterFn) -> Self:
        """Add condition for message forwarding."""
        self._filter_condition = condition
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
        self._filter_condition: AnyFilterFn | None = None
        self.active = True

    def __repr__(self):
        return f"TeamTalk({list(self)})"

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

    def when(self, condition: AnyFilterFn) -> Self:
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

    def __repr__(self):
        return f"TalkManager({self.owner})"

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
        transform: Callable[[Any], Any | Awaitable[Any]] | None = None,
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
        queue_strategy: Literal["concat", "latest", "buffer"] = "latest",
        transform: Callable[[Any], Any | Awaitable[Any]] | None = None,
        filter_condition: AnyFilterFn | None = None,
        stop_condition: AnyFilterFn | None = None,
        exit_condition: AnyFilterFn | None = None,
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
        transform: Callable[[Any], Any | Awaitable[Any]] | None = None,
        filter_condition: AnyFilterFn | None = None,
        stop_condition: AnyFilterFn | None = None,
        exit_condition: AnyFilterFn | None = None,
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
        queue_strategy: Literal["concat", "latest", "buffer"] = "latest",
        transform: Callable[[Any], Any | Awaitable[Any]] | None = None,
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
        other: AnyAgent[Any, Any] | Team[Any] | str,
        *,
        connection_type: ConnectionType = "run",
        priority: int = 0,
        delay: timedelta | None = None,
        queued: bool = False,
        queue_strategy: Literal["concat", "latest", "buffer"] = "latest",
        transform: Callable[[Any], Any | Awaitable[Any]] | None = None,
        filter_condition: AnyFilterFn | None = None,
        stop_condition: AnyFilterFn | None = None,
        exit_condition: AnyFilterFn | None = None,
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
        msg = "TalkManager routing message from %s to %d connections"
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
    from llmling_agent.talk.stats import TalkStats

    agent = Agent[None]("test_agent")
    agent_2 = Agent[None]("test_agent_2")
    agent_3 = Agent[None]("test_agent_3")
    agent_4 = Agent[None]("test_agent_3")
    _conn_1 = agent >> agent_2
    _conn_2 = agent >> agent_3
    _conn_3 = agent_2 >> agent_4
    print(agent.connections.get_connections(recursive=True))
    print(agent.connections.get_mermaid_diagram())
