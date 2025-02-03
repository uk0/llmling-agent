"""Manages message flow between agents/groups."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime
import inspect
from typing import TYPE_CHECKING, Any, Literal, Self

from psygnal import Signal
from typing_extensions import TypeVar

from llmling_agent.log import get_logger
from llmling_agent.messaging.messages import ChatMessage
from llmling_agent.models.events import ConnectionEvent, ConnectionEventType, EventData
from llmling_agent.talk.stats import AggregatedTalkStats, TalkStats


if TYPE_CHECKING:
    from collections.abc import Awaitable
    from datetime import timedelta
    import os

    import PIL.Image
    from toprompt import AnyPromptType

    from llmling_agent.common_types import AnyFilterFn, AnyTransformFn
    from llmling_agent.messaging.messagenode import MessageNode
    from llmling_agent.models.forward_targets import ConnectionType
    from llmling_agent.models.providers import ProcessorCallback

TContent = TypeVar("TContent")
QueueStrategy = Literal["concat", "latest", "buffer"]
logger = get_logger(__name__)


class Talk[TTransmittedData]:
    """Manages message flow between agents/groups."""

    @dataclass(frozen=True)
    class ConnectionProcessed:
        """Event emitted when a message flows through a connection."""

        message: ChatMessage
        source: MessageNode
        targets: list[MessageNode]
        queued: bool
        connection_type: ConnectionType
        timestamp: datetime = field(default_factory=datetime.now)

    # Original message "coming in"
    message_received = Signal(ChatMessage)
    # After any transformation (one for each message, not per target)
    message_forwarded = Signal(ChatMessage)
    # Comprehensive signal capturing all information about one "message handling process"
    connection_processed = Signal(ConnectionProcessed)

    def __init__(
        self,
        source: MessageNode,
        targets: Sequence[MessageNode],
        group: TeamTalk | None = None,
        *,
        name: str | None = None,
        connection_type: ConnectionType = "run",
        wait_for_connections: bool = False,
        priority: int = 0,
        delay: timedelta | None = None,
        queued: bool = False,
        queue_strategy: QueueStrategy = "latest",
        transform: AnyTransformFn[ChatMessage[TTransmittedData]] | None = None,
        filter_condition: AnyFilterFn | None = None,
        stop_condition: AnyFilterFn | None = None,
        exit_condition: AnyFilterFn | None = None,
    ):
        """Initialize talk connection.

        Args:
            source: Agent sending messages
            targets: Agents receiving messages
            group: Optional group this talk belongs to
            name: Optional name for this talk
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
        self.targets = list(targets)
        # Could perhaps better be an auto-inferring property
        self.name = name or f"{source.name}->{[t.name for t in targets]}"
        self.group = group
        self.priority = priority
        self.delay = delay
        self.active = True
        self.connection_type = connection_type
        self.wait_for_connections = wait_for_connections
        self.queued = queued
        self.queue_strategy = queue_strategy
        self._pending_messages: dict[str, list[ChatMessage[TTransmittedData]]] = {
            target.name: [] for target in targets
        }
        names = {t.name for t in targets}
        self._stats = TalkStats(source_name=source.name, target_names=names)
        self._transform = transform
        self._filter_condition = filter_condition
        self._stop_condition = stop_condition
        self._exit_condition = exit_condition

    def __repr__(self):
        targets = [t.name for t in self.targets]
        return f"<Talk({self.connection_type}) {self.source.name} -> {targets}>"

    def __rshift__(
        self,
        other: MessageNode[Any, Any]
        | ProcessorCallback[Any]
        | Sequence[MessageNode[Any, Any] | ProcessorCallback[Any]],
    ) -> Self:
        """Add another node as target to the connection or group.

        Example:
            connection >> other_agent  # Connect to single agent
            connection >> (agent2 & agent3)  # Connect to group
        """
        from llmling_agent.agent import Agent, StructuredAgent
        from llmling_agent.messaging.messagenode import MessageNode
        from llmling_agent.utils.inspection import has_return_type

        match other:
            case Callable():
                if has_return_type(other, str):
                    other = Agent.from_callback(other)
                else:
                    other = StructuredAgent.from_callback(other)
                self.targets.append(other)
            case Sequence():
                for o in other:
                    self.__rshift__(o)
            case MessageNode():
                self.targets.append(other)
            case _:
                msg = f"Invalid agent type: {type(other)}"
                raise TypeError(msg)
        # TODO: should return other?
        return self

    async def _evaluate_condition(
        self,
        condition: Callable[..., bool | Awaitable[bool]] | None,
        message: ChatMessage[Any],
        target: MessageNode,
        *,
        default_return: bool = False,
    ) -> bool:
        """Evaluate a condition with flexible parameter handling."""
        from llmling_agent.talk.registry import EventContext

        if not condition:
            return default_return
        ctx = EventContext(
            message=message,
            target=target,
            stats=self.stats,
            registry=self.source.context.pool.connection_registry
            if self.source.context and self.source.context.pool
            else None,
            talk=self,
        )
        result = condition(ctx)
        return await result if inspect.isawaitable(result) else result

    def on_event(
        self,
        event_type: ConnectionEventType,
        callback: Callable[[ConnectionEvent[TTransmittedData]], None | Awaitable[None]],
    ) -> Self:
        """Register callback for connection events."""

        async def wrapped_callback(event: EventData) -> None:
            if isinstance(event, ConnectionEvent) and event.event_type == event_type:
                result = callback(event)
                if inspect.isawaitable(result):
                    await result

        self.source._events.add_callback(wrapped_callback)
        return self

    async def _emit_connection_event(
        self,
        event_type: ConnectionEventType,
        message: ChatMessage[TTransmittedData] | None,
    ) -> None:
        event = ConnectionEvent(
            connection=self,
            source="connection",
            connection_name=self.name,
            event_type=event_type,
            message=message,
            timestamp=datetime.now(),
        )
        # Propagate to all event managers through registry
        if self.source.context and (pool := self.source.context.pool):
            for connection in pool.connection_registry.values():
                await connection.source._events.emit_event(event)

    async def _handle_message(
        self,
        message: ChatMessage[TTransmittedData],
        prompt: str | None = None,
    ) -> list[ChatMessage[Any]]:
        """Handle message forwarding based on connection configuration."""
        # 2. Early exit checks
        if not (self.active and (not self.group or self.group.active)):
            return []

        # 3. Check exit condition for any target
        for target in self.targets:
            # Exit if condition returns True
            if await self._evaluate_condition(self._exit_condition, message, target):
                raise SystemExit

        # 4. Check stop condition for any target
        for target in self.targets:
            # Stop if condition returns True
            if await self._evaluate_condition(self._stop_condition, message, target):
                self.disconnect()
                return []

        # 5. Transform if configured
        processed_message = message
        if self._transform:
            transformed = self._transform(message)
            if inspect.isawaitable(transformed):
                processed_message = await transformed
            else:
                processed_message = transformed

        # 6. First pass: Determine target list
        target_list: list[MessageNode] = [
            target
            for target in self.targets
            if await self._evaluate_condition(
                self._filter_condition,
                processed_message,
                target,
                default_return=True,
            )
        ]
        # 7. emit connection processed event
        self.connection_processed.emit(
            self.ConnectionProcessed(
                message=processed_message,
                source=self.source,
                targets=target_list,
                queued=self.queued,
                connection_type=self.connection_type,  # pyright: ignore
            )
        )
        # 8. if we have targets, update stats and emit message forwarded
        if target_list:
            messages = [*self._stats.messages, processed_message]
            self._stats = replace(self._stats, messages=messages)
            self.message_forwarded.emit(processed_message)

        # 9. Second pass: Actually process for each target
        responses: list[ChatMessage[Any]] = []
        for target in target_list:
            if self.queued:
                self._pending_messages[target.name].append(processed_message)
                continue
            if response := await self._process_for_target(
                processed_message, target, prompt
            ):
                responses.append(response)

        return responses

    async def _process_for_target(
        self,
        message: ChatMessage[Any],
        target: MessageNode,
        prompt: AnyPromptType | PIL.Image.Image | os.PathLike[str] | None = None,
    ) -> ChatMessage[Any] | None:
        """Process message for a single target."""
        from llmling_agent.agent import Agent, StructuredAgent
        from llmling_agent.delegation.base_team import BaseTeam

        match self.connection_type:
            case "run":
                prompts: list[AnyPromptType | PIL.Image.Image | os.PathLike[str]] = [
                    message
                ]
                if prompt:
                    prompts.append(prompt)
                return await target.run(*prompts)

            case "context":
                meta = {
                    "type": "forwarded_message",
                    "role": message.role,
                    "model": message.model,
                    "cost_info": message.cost_info,
                    "timestamp": message.timestamp.isoformat(),
                    "prompt": prompt,
                }

                async def add_context():
                    match target:
                        case BaseTeam():
                            # Use distribute for teams
                            await target.distribute(str(message.content), metadata=meta)
                        case Agent() | StructuredAgent():  # Agent case
                            # Use existing context message approach
                            target.conversation.add_context_message(
                                str(message.content),
                                source=message.name,
                                metadata=meta,
                            )

                if self.delay is not None or self.priority != 0:
                    target.run_background(
                        add_context(),
                        priority=self.priority,
                        delay=self.delay,
                    )
                else:
                    await add_context()
                return None

            case "forward":
                if self.delay is not None or self.priority != 0:

                    async def delayed_emit():
                        target.outbox.emit(message, prompt)

                    target.run_background(
                        delayed_emit(),
                        priority=self.priority,
                        delay=self.delay,
                    )
                else:
                    target.outbox.emit(message, prompt)
                return None

    async def trigger(
        self, prompt: AnyPromptType | PIL.Image.Image | os.PathLike[str] | None = None
    ) -> list[ChatMessage[TTransmittedData]]:
        """Process queued messages."""
        if not self._pending_messages:
            return []
        match self.queue_strategy:
            case "buffer":
                results: list[ChatMessage[TTransmittedData]] = []
                # Process each agent's queue
                for target in self.targets:
                    queue = self._pending_messages[target.name]
                    for message in queue:
                        if response := await self._process_for_target(
                            message, target, prompt
                        ):
                            results.append(response)  # noqa: PERF401
                    queue.clear()
                return results

            case "latest":
                results = []
                # Get latest message for each agent
                for target in self.targets:
                    queue = self._pending_messages[target.name]
                    if queue:
                        latest = queue[-1]
                        if response := await self._process_for_target(
                            latest, target, prompt
                        ):
                            results.append(response)
                        queue.clear()
                return results

            case "concat":
                results = []
                # Concat messages per agent
                for target in self.targets:
                    queue = self._pending_messages[target.name]
                    if not queue:
                        continue

                    base = queue[-1]
                    contents = [str(m.content) for m in queue]
                    meta = {
                        **base.metadata,
                        "merged_count": len(queue),
                        "queue_strategy": self.queue_strategy,
                    }
                    content = "\n\n".join(contents)
                    merged = replace(base, content=content, metadata=meta)  # type: ignore

                    if response := await self._process_for_target(merged, target, prompt):
                        results.append(response)
                    queue.clear()

                return results
            case _:
                msg = f"Invalid queue strategy: {self.queue_strategy}"
                raise ValueError(msg)

    def when(self, condition: AnyFilterFn) -> Self:
        """Add condition for message forwarding."""
        self._filter_condition = condition
        return self

    def transform[TNewData](
        self,
        transformer: Callable[
            [ChatMessage[TTransmittedData]],
            ChatMessage[TNewData] | Awaitable[ChatMessage[TNewData]],
        ],
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> Talk[TNewData]:
        """Chain a new transformation after existing ones.

        Args:
            transformer: Function to transform messages
            name: Optional name for debugging
            description: Optional description

        Returns:
            New Talk instance with chained transformation

        Example:
            ```python
            talk = (agent1 >> agent2)
                .transform(parse_json)      # str -> dict
                .transform(extract_values)  # dict -> list
            ```
        """
        new_talk = Talk[TNewData](
            source=self.source,
            targets=self.targets,
            connection_type=self.connection_type,  # type: ignore
        )

        if self._transform is not None:
            old_transform = self._transform

            async def chained_transform(
                data: ChatMessage[TTransmittedData],
            ) -> ChatMessage[TNewData]:
                intermediate = old_transform(data)
                if inspect.isawaitable(intermediate):
                    intermediate = await intermediate
                result = transformer(intermediate)
                if inspect.isawaitable(result):
                    return await result
                return result

            new_talk._transform = chained_transform  # type: ignore
        else:
            new_talk._transform = transformer  # type: ignore

        return new_talk

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


class TeamTalk[TTransmittedData](list["Talk | TeamTalk"]):
    """Group of connections with aggregate operations."""

    def __init__(
        self, talks: Sequence[Talk[TTransmittedData] | TeamTalk[TTransmittedData]]
    ):
        super().__init__(talks)
        self._filter_condition: AnyFilterFn | None = None
        self.active = True

    def __repr__(self):
        return f"TeamTalk({list(self)})"

    @property
    def targets(self) -> list[MessageNode]:
        """Get all targets from all connections."""
        return [t for talk in self for t in talk.targets]

    async def _handle_message(self, message: ChatMessage[Any], prompt: str | None = None):
        for talk in self:
            await talk._handle_message(message, prompt)

    async def trigger(
        self, prompt: AnyPromptType | PIL.Image.Image | os.PathLike[str] | None = None
    ) -> list[ChatMessage]:
        messages = []
        for talk in self:
            messages.extend(await talk.trigger(prompt))
        return messages

    @classmethod
    def from_nodes(
        cls,
        agents: Sequence[MessageNode],
        targets: list[MessageNode] | None = None,
    ) -> Self:
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
    def stats(self) -> AggregatedTalkStats:
        """Get aggregated statistics for all connections."""
        return AggregatedTalkStats(stats=[talk.stats for talk in self])

    def when(self, condition: AnyFilterFn) -> Self:
        """Add condition to all connections in group."""
        for talk in self:
            talk.when(condition)
        return self

    def disconnect(self):
        """Disconnect all connections in group."""
        for talk in self:
            talk.disconnect()
