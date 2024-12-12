"""History management functionality for LLMling agent."""

from datetime import datetime
from collections.abc import Sequence

from llmling_agent.history.filters import parse_time_period, validate_group_by
from llmling_agent.history.formatters import (
    format_conversation,
    format_message,
    format_stats,
)
from llmling_agent.history.models import (
    ConversationData,
    GroupBy,
    MessageData,
    QueryFilters,
    StatsFilters,
)
from llmling_agent.history.queries import get_conversations
from llmling_agent.history.stats import get_conversation_stats


__all__ = [
    "ConversationData",
    "GroupBy",
    "MessageData",
    "QueryFilters",
    "StatsFilters",
    "format_conversation",
    "format_message",
    "format_stats",
    "get_conversation_stats",
    "get_conversations",
    "parse_time_period",
    "validate_group_by",
]


def get_filtered_conversations(
    agent_name: str | None = None,
    since: datetime | None = None,
    query: str | None = None,
    model: str | None = None,
    limit: int | None = None,
) -> Sequence[ConversationData]:
    """High-level function to get filtered conversations.

    Combines query and formatting logic for common use case.
    """
    filters = QueryFilters(
        agent_name=agent_name,
        since=since,
        query=query,
        model=model,
        limit=limit,
    )
    conversations = get_conversations(filters)
    return [format_conversation(conv, msgs) for conv, msgs in conversations]
