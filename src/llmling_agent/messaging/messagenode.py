"""Base class for message processing nodes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Coroutine, Sequence
from typing import TYPE_CHECKING, Any, TypeVar, overload

from psygnal import Signal

from llmling_agent.utils.inspection import has_return_type
from llmling_agent.utils.tasks import TaskManagerMixin


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from datetime import timedelta
    import os

    import PIL.Image
    from toprompt import AnyPromptType

    from llmling_agent.common_types import AnyTransformFn, AsyncFilterFn
    from llmling_agent.models.forward_targets import ConnectionType
    from llmling_agent.models.messages import ChatMessage
    from llmling_agent.models.providers import ProcessorCallback
    from llmling_agent.talk import QueueStrategy, Talk, TeamTalk
    from llmling_agent.talk.talk import AnyTeamOrAgent


NodeType = TypeVar("NodeType", bound="MessageNode")
TResult = TypeVar("TResult")


class MessageNode[TDeps, TResult](TaskManagerMixin, ABC):
    """Base class for all message processing nodes."""

    outbox = Signal(object)  # ChatMessage
    """Signal emitted when node produces a message."""

    def __init__(self, name: str | None = None):
        """Initialize message node."""
        super().__init__()
        from llmling_agent.agent.connection import ConnectionManager

        self._name = name or self.__class__.__name__
        self.connections = ConnectionManager(self)

    @property
    def name(self) -> str:
        """Get agent name."""
        return self._name or "llmling-agent"

    @name.setter
    def name(self, value: str):
        self._name = value

    @overload
    def connect_to(
        self,
        target: AnyTeamOrAgent[Any, Any] | ProcessorCallback[Any],
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
    ) -> Talk[Any]: ...

    @overload
    def connect_to(
        self,
        target: Sequence[AnyTeamOrAgent[Any, Any] | ProcessorCallback[Any]],
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
    ) -> TeamTalk: ...

    def connect_to(
        self,
        target: AnyTeamOrAgent[Any, Any]
        | ProcessorCallback[Any]
        | Sequence[AnyTeamOrAgent[Any, Any] | ProcessorCallback[Any]],
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
            delay=delay,
            queued=queued,
            queue_strategy=queue_strategy,
            transform=transform,
            filter_condition=filter_condition,
            stop_condition=stop_condition,
            exit_condition=exit_condition,
        )

    async def run(
        self,
        *prompts: AnyPromptType | PIL.Image.Image | os.PathLike[str],
        wait_for_connections: bool | None = None,
        **kwargs: Any,
    ) -> ChatMessage[TResult]:
        """Execute node with prompts and handle message routing.

        Args:
            prompts: Input prompts
            wait_for_connections: Whether to wait for forwarded messages
            **kwargs: Additional arguments for _run
        """
        message = await self._run(*prompts, **kwargs)
        await self.connections.route_message(
            message,
            wait=wait_for_connections,
        )
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
