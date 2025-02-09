"""Event source implementation."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from llmling_agent.messaging.messageemitter import MessageEmitter


if TYPE_CHECKING:
    from collections.abc import Coroutine

    from llmling_agent.messaging.messages import ChatMessage
    from llmling_agent.models.events import EventSourceConfig


class EventNode[TEventData](MessageEmitter[None, TEventData]):
    """Base class for event sources.

    An event source monitors for events and emits them as messages.
    Generically typed with the type of event data it produces.
    """

    def __init__(
        self,
        config: EventSourceConfig,
        name: str | None = None,
    ) -> None:
        """Initialize event source.

        Args:
            config: Configuration for this event source
            name: Optional name override
        """
        super().__init__(name=name or f"{config.type}_{config.name}")
        self.config = config
        self._running = False

    async def start(self) -> None:
        """Start monitoring for events."""
        self._running = True
        try:
            await self._monitor()
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
