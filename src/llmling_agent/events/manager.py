"""Event manager for handling multiple event sources."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Self

from llmling_agent.events.sources import (
    EmailConfig,
    EventConfig,
    EventData,
    FileWatchConfig,
    WebhookConfig,
)
from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from llmling_agent.agent import AnyAgent
    from llmling_agent_events.base import EventSource

logger = get_logger(__name__)


type EventCallback = Callable[[EventData], None | Awaitable[None]]


class EventManager:
    """Manages multiple event sources and their lifecycles."""

    def __init__(
        self,
        agent: AnyAgent[Any, Any],
        enable_events: bool = True,
        auto_run: bool = True,
    ):
        """Initialize event manager.

        Args:
            agent: Agent to manage events for
            enable_events: Whether to enable event processing
            auto_run: Whether to automatically call run() for event callbacks
        """
        self.agent = agent
        self.enabled = enable_events
        self._sources: dict[str, EventSource] = {}
        self._callbacks: list[EventCallback] = []
        self.auto_run = auto_run

    async def _default_handler(self, event: EventData) -> None:
        """Default event handler that converts events to agent runs."""
        if prompt := event.to_prompt():  # Only run if event provides a prompt
            await self.agent.run(prompt)

    async def add_callback(self, callback: EventCallback) -> None:
        """Register an event callback."""
        self._callbacks.append(callback)

    async def remove_callback(self, callback: EventCallback) -> None:
        """Remove a previously registered callback."""
        self._callbacks.remove(callback)

    async def emit_event(self, event: EventData) -> None:
        """Emit event to all callbacks and optionally handle via agent."""
        if not self.enabled:
            return

        # Run custom callbacks
        for callback in self._callbacks:
            try:
                result = callback(event)
                if isinstance(result, Awaitable):
                    await result
            except Exception:
                logger.exception("Error in event callback %r", callback.__name__)

        # Run default handler if enabled
        if self.auto_run:
            try:
                prompt = event.to_prompt()
                if prompt:
                    await self.agent.run(prompt)
            except Exception:
                logger.exception("Error in default event handler")

    def create_source(self, config: EventConfig) -> EventSource:
        """Create an event source from configuration.

        Args:
            config: Event source configuration

        Returns:
            Configured event source instance

        Raises:
            ValueError: If source type is unknown or disabled
        """
        if not config.enabled:
            msg = f"Source {config.name} is disabled"
            raise ValueError(msg)

        match config:
            case FileWatchConfig():
                from llmling_agent_events.file_watcher import FileSystemEventSource

                return FileSystemEventSource(config)
            case WebhookConfig():
                from llmling_agent_events.webhook_watcher import WebhookEventSource

                return WebhookEventSource(config)

            case EmailConfig():
                from llmling_agent_events.email_watcher import EmailEventSource

                return EmailEventSource(config)
            case _:
                msg = f"Unknown event source type: {config.type}"
                raise ValueError(msg)

    async def add_source(self, config: EventConfig):
        """Add and start a new event source.

        Args:
            config: Event source configuration

        Raises:
            ValueError: If source already exists or is invalid
        """
        if not self.enabled:
            msg = "Event processing disabled, not adding source: %s"
            logger.warning(msg, config.name)
            return

        if config.name in self._sources:
            msg = f"Event source already exists: {config.name}"
            raise ValueError(msg)

        try:
            source = self.create_source(config)
            await source.connect()
            self._sources[config.name] = source

            # Start processing events
            name = f"event_processor_{config.name}"
            self.agent.create_task(self._process_events(source), name=name)
            logger.debug("Added event source: %s", config.name)

        except Exception as e:
            msg = f"Failed to add event source {config.name}"
            logger.exception(msg)
            raise RuntimeError(msg) from e

    async def remove_source(self, name: str):
        """Stop and remove an event source.

        Args:
            name: Name of source to remove
        """
        if source := self._sources.pop(name, None):
            await source.disconnect()
            logger.debug("Removed event source: %s", name)

    async def _process_events(self, source: EventSource):
        """Process events from a source.

        Args:
            source: Event source to process
        """
        try:
            # Get the async iterator from the coroutine
            async for event in source.events():
                if not self.enabled:
                    break
                await self.emit_event(event)

        except asyncio.CancelledError:
            logger.debug("Event processing cancelled")
            raise

        except Exception:
            logger.exception("Error processing events")

    async def cleanup(self):
        """Clean up all event sources and tasks."""
        self.enabled = False

        for name in list(self._sources):
            await self.remove_source(name)

    async def __aenter__(self) -> Self:
        """Allow using manager as async context manager."""
        return self

    async def __aexit__(self, *exc: object):
        """Clean up when exiting context."""
        await self.cleanup()
