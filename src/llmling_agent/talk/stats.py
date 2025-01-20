"""Manages message flow between agents/groups."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from functools import cached_property
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from llmling_agent.models.messages import ChatMessage, FormatStyle


@dataclass(frozen=True)
class TalkStats:
    """Statistics for a single connection."""

    source_name: str | None
    target_names: set[str]
    messages: list[ChatMessage[Any]] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)

    @property
    def message_count(self) -> int:
        """Number of messages transmitted."""
        return len(self.messages)

    @property
    def last_message_time(self) -> datetime | None:
        """When the last message was sent."""
        return self.messages[-1].timestamp if self.messages else None

    @property
    def token_count(self) -> int:
        """Total tokens used."""
        return sum(
            msg.cost_info.token_usage["total"] for msg in self.messages if msg.cost_info
        )

    @property
    def byte_count(self) -> int:
        """Total bytes transmitted."""
        return sum(len(str(msg.content).encode()) for msg in self.messages)

    def format(
        self,
        style: FormatStyle = "simple",
        **kwargs: Any,
    ) -> str:
        """Format the conversation that happened on this connection."""
        return "\n".join(msg.format(style, **kwargs) for msg in self.messages)

    @property
    def total_cost(self) -> float:
        """Total cost in USD."""
        return sum(
            float(msg.cost_info.total_cost) for msg in self.messages if msg.cost_info
        )


@dataclass
class TeamTalkStats:
    """Statistics aggregated from multiple connections."""

    stats: list[TalkStats | TeamTalkStats] = field(default_factory=list)

    # def __init__(self, stats: list[TalkStats | TeamTalkStats]):
    #     self.stats = stats

    @property
    def messages(self) -> list[ChatMessage[Any]]:
        """All messages across all connections, flattened."""
        return [msg for stat in self.stats for msg in stat.messages]

    @property
    def message_count(self) -> int:
        """Total messages across all connections."""
        return len(self.messages)

    @property
    def start_time(self) -> datetime:
        """Total messages across all connections."""
        return self.stats[0].start_time

    @property
    def num_connections(self) -> int:
        """Number of active connections."""
        return len(self.stats)

    @property
    def token_count(self) -> int:
        """Total tokens across all connections."""
        return sum(
            msg.cost_info.token_usage["total"] for msg in self.messages if msg.cost_info
        )

    @property
    def total_cost(self) -> float:
        """Total cost in USD."""
        return sum(
            float(msg.cost_info.total_cost) for msg in self.messages if msg.cost_info
        )

    @property
    def byte_count(self) -> int:
        """Total bytes transmitted."""
        return sum(len(str(msg.content).encode()) for msg in self.messages)

    @property
    def last_message_time(self) -> datetime | None:
        """Most recent message time."""
        if not self.messages:
            return None
        return max(msg.timestamp for msg in self.messages)

    def format(
        self,
        style: FormatStyle = "simple",
        **kwargs: Any,
    ) -> str:
        """Format all conversations in the team."""
        return "\n".join(msg.format(style, **kwargs) for msg in self.messages)

    @cached_property
    def source_names(self) -> set[str]:
        """Set of unique source names recursively."""

        def _collect_source_names(stat: TalkStats | TeamTalkStats) -> set[str]:
            """Recursively collect source names."""
            if isinstance(stat, TalkStats):
                return {stat.source_name} if stat.source_name else set()
            # It's a TeamTalkStats, recurse
            names = set()
            for s in stat.stats:
                names.update(_collect_source_names(s))
            return names

        names = set()
        for stat in self.stats:
            names.update(_collect_source_names(stat))
        return names

    @cached_property
    def target_names(self) -> set[str]:
        """Set of all target names across connections."""
        return {name for s in self.stats for name in s.target_names}
