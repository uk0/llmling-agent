"""Base class for message processing nodes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Coroutine, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal, Self, TypeVar, overload
from uuid import uuid4

from psygnal import Signal

from llmling_agent.mcp_server.manager import MCPManager
from llmling_agent.messaging.messages import ChatMessage
from llmling_agent.prompts.convert import convert_prompts
from llmling_agent.talk.stats import (
    AggregatedMessageStats,
    AggregatedTalkStats,
    MessageStats,
)
from llmling_agent.utils.inspection import has_return_type
from llmling_agent.utils.tasks import TaskManagerMixin


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from datetime import timedelta
    import os
    from types import TracebackType

    import PIL.Image
    from toprompt import AnyPromptType

    from llmling_agent.common_types import AnyTransformFn, AsyncFilterFn
    from llmling_agent.delegation.pool import AgentPool
    from llmling_agent.models.agents import AgentsManifest
    from llmling_agent.models.forward_targets import ConnectionType
    from llmling_agent.models.mcp_server import MCPServerConfig
    from llmling_agent.models.nodes import NodeConfig
    from llmling_agent.models.providers import ProcessorCallback
    from llmling_agent.talk import QueueStrategy, Talk, TeamTalk


NodeType = TypeVar("NodeType", bound="MessageNode")
TResult = TypeVar("TResult")


@dataclass(kw_only=True)
class NodeContext[TDeps]:
    node_name: str
    """Name of the current node."""

    pool: AgentPool | None = None
    """The agent pool the node is part of."""

    config: NodeConfig
    """Node configuration."""

    definition: AgentsManifest
    """Complete agent definition with all configurations."""

    current_prompt: str | None = None
    """Current prompt text for the agent."""

    in_async_context: bool = False
    """Whether we're running in an async context."""

    @property
    def node(self) -> MessageNode[TDeps, Any]:
        """Get the agent instance from the pool."""
        assert self.pool, "No agent pool available"
        assert self.node_name, "No agent name available"
        return self.pool[self.node_name]


class MessageNode[TDeps, TResult](TaskManagerMixin, ABC):
    """Base class for all message processing nodes."""

    outbox = Signal(object)  # ChatMessage
    """Signal emitted when node produces a message."""

    message_received = Signal(ChatMessage)
    """Signal emitted when node receives a message."""

    message_sent = Signal(ChatMessage)
    """Signal emitted when node creates a message."""

    def __init__(
        self,
        name: str | None = None,
        description: str | None = None,
        context: NodeContext | None = None,
        mcp_servers: Sequence[str | MCPServerConfig] | None = None,
    ):
        """Initialize message node."""
        super().__init__()
        from llmling_agent.messaging.connection_manager import ConnectionManager
        from llmling_agent.messaging.event_manager import EventManager

        self._name = name or self.__class__.__name__
        self.description = description
        self.connections = ConnectionManager(self)
        self._events = EventManager(self, enable_events=True)
        self.mcp = MCPManager(servers=mcp_servers or [], context=context)

    async def __aenter__(self) -> Self:
        """Initialize base message node."""
        try:
            await self._events.__aenter__()
            await self.mcp.__aenter__()
        except Exception as e:
            await self.__aexit__(type(e), e, e.__traceback__)
            msg = f"Failed to initialize {self.name}"
            raise RuntimeError(msg) from e
        else:
            return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Clean up base resources."""
        await self._events.cleanup()
        await self.mcp.__aexit__(exc_type, exc_val, exc_tb)
        await self.cleanup_tasks()

    @property
    @abstractmethod
    def stats(self) -> MessageStats | AggregatedMessageStats:
        """Get stats for this node."""
        raise NotImplementedError

    @property
    def connection_stats(self) -> AggregatedTalkStats:
        """Get stats for all active connections of this node."""
        return AggregatedTalkStats(
            stats=[talk.stats for talk in self.connections.get_connections()]
        )

    @property
    def context(self) -> NodeContext:
        """Get node context."""
        raise NotImplementedError

    @property
    def name(self) -> str:
        """Get agent name."""
        return self._name or "llmling-agent"

    @name.setter
    def name(self, value: str):
        self._name = value

    @overload
    def __rshift__(
        self, other: MessageNode[Any, Any] | ProcessorCallback[Any]
    ) -> Talk[TResult]: ...

    @overload
    def __rshift__(
        self, other: Sequence[MessageNode[Any, Any] | ProcessorCallback[Any]]
    ) -> TeamTalk[TResult]: ...

    def __rshift__(
        self,
        other: MessageNode[Any, Any]
        | ProcessorCallback[Any]
        | Sequence[MessageNode[Any, Any] | ProcessorCallback[Any]],
    ) -> Talk[Any] | TeamTalk[Any]:
        """Connect agent to another agent or group.

        Example:
            agent >> other_agent  # Connect to single agent
            agent >> (agent2 & agent3)  # Connect to group
            agent >> "other_agent"  # Connect by name (needs pool)
        """
        return self.connect_to(other)

    @overload
    def connect_to(
        self,
        target: MessageNode[Any, Any] | ProcessorCallback[Any],
        *,
        queued: Literal[True],
        queue_strategy: Literal["concat"],
    ) -> Talk[str]: ...

    @overload
    def connect_to(
        self,
        target: MessageNode[Any, Any] | ProcessorCallback[Any],
        *,
        connection_type: ConnectionType = "run",
        name: str | None = None,
        priority: int = 0,
        delay: timedelta | None = None,
        queued: bool = False,
        queue_strategy: QueueStrategy = "latest",
        transform: AnyTransformFn | None = None,
        filter_condition: AsyncFilterFn | None = None,
        stop_condition: AsyncFilterFn | None = None,
        exit_condition: AsyncFilterFn | None = None,
    ) -> Talk[TResult]: ...

    @overload
    def connect_to(
        self,
        target: Sequence[MessageNode[Any, Any] | ProcessorCallback[Any]],
        *,
        queued: Literal[True],
        queue_strategy: Literal["concat"],
    ) -> TeamTalk[str]: ...

    @overload
    def connect_to(
        self,
        target: Sequence[MessageNode[Any, TResult] | ProcessorCallback[TResult]],
        *,
        connection_type: ConnectionType = "run",
        name: str | None = None,
        priority: int = 0,
        delay: timedelta | None = None,
        queued: bool = False,
        queue_strategy: QueueStrategy = "latest",
        transform: AnyTransformFn | None = None,
        filter_condition: AsyncFilterFn | None = None,
        stop_condition: AsyncFilterFn | None = None,
        exit_condition: AsyncFilterFn | None = None,
    ) -> TeamTalk[TResult]: ...

    @overload
    def connect_to(
        self,
        target: Sequence[MessageNode[Any, Any] | ProcessorCallback[Any]],
        *,
        connection_type: ConnectionType = "run",
        name: str | None = None,
        priority: int = 0,
        delay: timedelta | None = None,
        queued: bool = False,
        queue_strategy: QueueStrategy = "latest",
        transform: AnyTransformFn | None = None,
        filter_condition: AsyncFilterFn | None = None,
        stop_condition: AsyncFilterFn | None = None,
        exit_condition: AsyncFilterFn | None = None,
    ) -> TeamTalk: ...

    def connect_to(
        self,
        target: MessageNode[Any, Any]
        | ProcessorCallback[Any]
        | Sequence[MessageNode[Any, Any] | ProcessorCallback[Any]],
        *,
        connection_type: ConnectionType = "run",
        name: str | None = None,
        priority: int = 0,
        delay: timedelta | None = None,
        queued: bool = False,
        queue_strategy: QueueStrategy = "latest",
        transform: AnyTransformFn | None = None,
        filter_condition: AsyncFilterFn | None = None,
        stop_condition: AsyncFilterFn | None = None,
        exit_condition: AsyncFilterFn | None = None,
    ) -> Talk[Any] | TeamTalk:
        """Create connection(s) to target(s)."""
        # Handle callable case
        from llmling_agent.agent import Agent, StructuredAgent
        from llmling_agent.delegation.base_team import BaseTeam

        if callable(target):
            if has_return_type(target, str):
                target = Agent.from_callback(target)
            else:
                target = StructuredAgent.from_callback(target)
        # we are explicit here just to make disctinction clear, we only want sequences
        # of message units
        if isinstance(target, Sequence) and not isinstance(target, BaseTeam):
            targets: list[Agent | StructuredAgent] = []
            for t in target:
                match t:
                    case _ if callable(t):
                        if has_return_type(t, str):
                            targets.append(Agent.from_callback(t))
                        else:
                            targets.append(StructuredAgent.from_callback(t))
                    case Agent() | StructuredAgent():
                        targets.append(t)
                    case _:
                        msg = f"Invalid agent type: {type(t)}"
                        raise TypeError(msg)
        else:
            targets = target  # type: ignore
        return self.connections.create_connection(
            self,
            targets,
            connection_type=connection_type,
            priority=priority,
            name=name,
            delay=delay,
            queued=queued,
            queue_strategy=queue_strategy,
            transform=transform,
            filter_condition=filter_condition,
            stop_condition=stop_condition,
            exit_condition=exit_condition,
        )

    async def disconnect_all(self):
        """Disconnect from all nodes."""
        for target in list(self.connections.get_targets()):
            self.stop_passing_results_to(target)

    def stop_passing_results_to(self, other: MessageNode):
        """Stop forwarding results to another node."""
        self.connections.disconnect(other)

    async def run(
        self,
        *prompt: AnyPromptType | PIL.Image.Image | os.PathLike[str] | ChatMessage,
        wait_for_connections: bool | None = None,
        store_history: bool = True,
        **kwargs: Any,
    ) -> ChatMessage[TResult]:
        """Execute node with prompts and handle message routing.

        Args:
            prompt: Input prompts
            wait_for_connections: Whether to wait for forwarded messages
            store_history: Whether to store in conversation history
            **kwargs: Additional arguments for _run
        """
        from llmling_agent import Agent, StructuredAgent

        if len(prompt) == 1 and isinstance(prompt[0], ChatMessage):
            user_msg = prompt[0]
            prompts = await convert_prompts([user_msg.content])
            # Update received message's chain to show it came through its source
            user_msg = user_msg.forwarded(prompt[0])
            conversation_id = user_msg.conversation_id

            final_prompt = "\n\n".join(str(p) for p in prompts)
        else:
            prompts = await convert_prompts(prompt)
            final_prompt = "\n\n".join(str(p) for p in prompts)
            conversation_id = str(uuid4())
            # use format_prompts?
            user_msg = ChatMessage[str](
                content=final_prompt,
                role="user",
                conversation_id=conversation_id,
            )
        self.message_received.emit(user_msg)
        self.context.current_prompt = final_prompt
        message = await self._run(*prompts, store_history=store_history, **kwargs)

        # For chain processing, update the response's chain
        if len(prompt) == 1 and isinstance(prompt[0], ChatMessage):
            message = message.forwarded(prompt[0])

        # Set conversation_id on response message
        message = replace(message, conversation_id=conversation_id)

        if store_history and isinstance(self, Agent | StructuredAgent):
            self.conversation.add_chat_messages([user_msg, message])
        self.message_sent.emit(message)
        await self.connections.route_message(message, wait=wait_for_connections)
        return message

    @abstractmethod
    def _run(
        self,
        *prompts: Any,
        **kwargs: Any,
    ) -> Coroutine[None, None, ChatMessage[TResult]]:
        """Implementation-specific run logic."""

    @abstractmethod
    def run_iter(
        self,
        *prompts: Any,
        **kwargs: Any,
    ) -> AsyncIterator[ChatMessage[Any]]:
        """Yield messages during execution."""
