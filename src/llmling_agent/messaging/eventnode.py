"""Event source implementation."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from llmling_agent.messaging.messageemitter import MessageEmitter


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Coroutine

    from llmling_agent.messaging.context import NodeContext
    from llmling_agent.messaging.messages import ChatMessage


class Event[TEventData]:
    """Base class for event implementations.

    Handles monitoring for and converting specific types of events.
    Generically typed with the type of event data produced.
    """

    @abstractmethod
    def create_monitor(self) -> AsyncGenerator[Any, None]:
        """Create async generator that yields raw event data.

        Yields:
            Raw event data that will be passed to convert_data
        """
        raise NotImplementedError

    @abstractmethod
    def convert_data(self, raw_data: Any) -> TEventData:
        """Convert raw event data to typed event data.

        Args:
            raw_data: Data from create_monitor

        Returns:
            Typed event data
        """
        raise NotImplementedError


class EventNode[TEventData](MessageEmitter[None, TEventData]):
    """Base class for event sources.

    An event source monitors for events and emits them as messages.
    Generically typed with the type of event data it produces.
    """

    def __init__(
        self,
        event: Event[TEventData],
        name: str | None = None,
        context: NodeContext | None = None,
        description: str | None = None,
    ) -> None:
        """Initialize event node.

        Args:
            event: Event implementation
            name: Optional name for this node
            context: Optional node context
            description: Optional description
        """
        super().__init__(name=name, context=context, description=description)
        self.event = event
        self._running = False

    async def start(self) -> None:
        """Start monitoring for events."""
        self._running = True
        try:
            async for data in self.event.create_monitor():
                if not self._running:
                    break
                await self.run(data)
        finally:
            self._running = False

    async def stop(self) -> None:
        """Stop monitoring for events."""
        self._running = False

    @abstractmethod
    async def _monitor(self) -> None:
        """Implementation-specific event monitoring.

        Should continuously monitor for events and call run() when they occur.
        Must respect self._running flag for clean shutdown.
        """
        raise NotImplementedError

    @abstractmethod
    def _run(
        self,
        *prompts: Any,
        **kwargs: Any,
    ) -> Coroutine[None, None, ChatMessage[TEventData]]:
        """Implementation-specific run logic."""
