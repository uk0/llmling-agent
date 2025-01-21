"""Event sources for LLMling agent."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from llmling_agent.events.sources import EventData


class EventSource(ABC):
    """Base class for event sources."""

    @abstractmethod
    async def connect(self):
        """Initialize connection to event source."""

    @abstractmethod
    async def disconnect(self):
        """Close connection to event source."""

    @abstractmethod
    def events(self) -> AsyncGenerator[EventData, None]:
        """Get event iterator.

        Returns:
            AsyncIterator yielding events from this source

        Note: This is a coroutine that returns an AsyncIterator
        """
        raise NotImplementedError
