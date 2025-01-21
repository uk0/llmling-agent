"""Time-based event source for scheduling agent actions."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING
import zoneinfo

from croniter import croniter

from llmling_agent.events.sources import EventData, TimeEventConfig
from llmling_agent_events.base import EventSource


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


@dataclass(frozen=True)
class TimeEvent(EventData):
    """Time-based event."""

    schedule: str
    """Cron expression that triggered this event."""

    def to_prompt(self) -> str:
        """Format scheduled event."""
        return f"Scheduled task triggered by {self.schedule}"


class TimeEventSource(EventSource):
    """Generates events based on cron schedule."""

    def __init__(self, config: TimeEventConfig):
        self.config = config
        self._stop_event = asyncio.Event()
        self._tz = zoneinfo.ZoneInfo(config.timezone) if config.timezone else None

    async def connect(self):
        """Validate cron expression."""
        try:
            now = datetime.now(self._tz)
            croniter(self.config.schedule, now)
        except Exception as e:
            msg = f"Invalid cron expression: {e}"
            raise ValueError(msg) from e

    async def disconnect(self):
        """Stop event generation."""
        self._stop_event.set()

    async def events(self) -> AsyncGenerator[EventData, None]:
        """Generate events based on schedule."""
        while not self._stop_event.is_set():
            now = datetime.now(self._tz)
            cron = croniter(self.config.schedule, now)
            next_run = cron.get_next(datetime)

            # Sleep until next run
            delay = (next_run - now).total_seconds()
            if delay > 0:
                try:
                    await asyncio.sleep(delay)
                except asyncio.CancelledError:
                    break

            # Skip if we missed the schedule and skip_missed is True
            if self.config.skip_missed:
                current = datetime.now(self._tz)
                if (current - next_run).total_seconds() > 1:
                    continue

            yield TimeEvent(source=self.config.name, schedule=self.config.schedule)
