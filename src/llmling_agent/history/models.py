from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypedDict


if TYPE_CHECKING:
    from datetime import datetime

    from llmling_agent.models import TokenUsage


GroupBy = Literal["agent", "model", "hour", "day"]


class MessageData(TypedDict):
    """Formatted message data."""

    role: str
    content: str
    timestamp: str
    model: str | None
    name: str | None
    token_usage: TokenUsage | None
    cost: float | None
    response_time: float | None


class ConversationData(TypedDict):
    """Formatted conversation data."""

    id: str
    agent: str
    start_time: str
    messages: list[MessageData]
    token_usage: TokenUsage | None


@dataclass
class QueryFilters:
    """Filters for conversation queries."""

    agent_name: str | None = None
    since: datetime | None = None
    query: str | None = None
    model: str | None = None
    limit: int | None = None


@dataclass
class StatsFilters:
    """Filters for statistics queries."""

    cutoff: datetime
    group_by: GroupBy
    agent_name: str | None = None
