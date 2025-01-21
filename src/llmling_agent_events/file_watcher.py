"""Event sources for LLMling agent."""

from __future__ import annotations

from asyncio import Event
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from watchfiles import Change, awatch
from watchfiles.filters import DefaultFilter

from llmling_agent.events.sources import EventData
from llmling_agent_events.base import EventSource


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator

    from watchfiles.main import FileChange

    from llmling_agent.events.sources import FileWatchConfig


ChangeType = Literal["added", "modified", "deleted"]

CHANGE_TO_TYPE: dict[Change, ChangeType] = {
    Change.added: "added",
    Change.modified: "modified",
    Change.deleted: "deleted",
}


@dataclass(frozen=True, kw_only=True)
class FileEvent(EventData):
    """File system event."""

    path: str
    type: ChangeType

    def to_prompt(self) -> str:
        return f"File {self.type}: {self.path}"


class ExtensionFilter:
    """Filter for specific file extensions."""

    def __init__(self, extensions: list[str], ignore_paths: list[str] | None = None):
        """Initialize filter.

        Args:
            extensions: File extensions to watch (e.g. ['.py', '.md'])
            ignore_paths: Paths to ignore
        """
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
        self._stop_event: Event | None = None

    async def connect(self):
        """Set up watchfiles watcher."""
        if not self.config.paths:
            msg = "No paths specified to watch"
            raise ValueError(msg)

        self._stop_event = Event()

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
        if not watch:
            msg = "Source not connected"
            raise RuntimeError(msg)

        async for changes in watch:
            for change, path in changes:
                if change not in CHANGE_TO_TYPE:
                    continue
                typ = CHANGE_TO_TYPE[change]
                yield FileEvent.create(source=self.config.name, path=str(path), type=typ)
