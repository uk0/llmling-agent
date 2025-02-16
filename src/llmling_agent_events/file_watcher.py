"""File event source."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from llmling_agent.messaging.events import ChangeType, FileEventData
from llmling_agent_events.base import EventSource


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator

    from watchfiles import Change
    from watchfiles.main import FileChange

    from llmling_agent.messaging.events import EventData
    from llmling_agent_config.events import FileWatchConfig


class ExtensionFilter:
    """Filter for specific file extensions."""

    def __init__(self, extensions: list[str], ignore_paths: list[str] | None = None):
        """Initialize filter.

        Args:
            extensions: File extensions to watch (e.g. ['.py', '.md'])
            ignore_paths: Paths to ignore
        """
        from watchfiles.filters import DefaultFilter

        self.extensions = tuple(
            ext if ext.startswith(".") else f".{ext}" for ext in extensions
        )
        self._default_filter = DefaultFilter(ignore_paths=ignore_paths)

    def __call__(self, change: Change, path: str) -> bool:
        """Check if file should be watched."""
        return path.endswith(self.extensions) and self._default_filter(change, path)


class FileSystemEventSource(EventSource):
    """Watch file system changes using watchfiles."""

    def __init__(self, config: FileWatchConfig):
        """Initialize file system watcher.

        Args:
            config: File watch configuration
        """
        self.config = config
        self._watch: AsyncIterator[set[FileChange]] | None = None
        self._stop_event: asyncio.Event | None = None

    async def connect(self):
        """Set up watchfiles watcher."""
        if not self.config.paths:
            msg = "No paths specified to watch"
            raise ValueError(msg)

        from watchfiles.main import awatch

        self._stop_event = asyncio.Event()

        # Create filter from extensions if provided
        watch_filter = None
        if self.config.extensions:
            to_ignore = self.config.ignore_paths
            watch_filter = ExtensionFilter(self.config.extensions, ignore_paths=to_ignore)

        self._watch = awatch(
            *self.config.paths,
            watch_filter=watch_filter,
            debounce=self.config.debounce,
            stop_event=self._stop_event,
            recursive=self.config.recursive,
        )

    async def disconnect(self):
        """Stop watchfiles watcher."""
        if self._stop_event:
            self._stop_event.set()
        self._watch = None
        self._stop_event = None

    async def events(self) -> AsyncGenerator[EventData, None]:
        """Get file system events."""
        watch = self._watch
        from watchfiles import Change

        if not watch:
            msg = "Source not connected"
            raise RuntimeError(msg)
        change_to_type: dict[Change, ChangeType] = {
            Change.added: "added",
            Change.modified: "modified",
            Change.deleted: "deleted",
        }
        async for changes in watch:
            for change, path in changes:
                if change not in change_to_type:
                    continue
                typ = change_to_type[change]
                yield FileEventData.create(
                    source=self.config.name, path=str(path), type=typ
                )
