"""Utility functions for Supabase provider."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from postgrest.types import CountMethod

from llmling_agent.messaging.messages import ChatMessage, TokenCost


if TYPE_CHECKING:
    from postgrest._async.request_builder import (
        AsyncRequestBuilder,
        AsyncSelectRequestBuilder,
    )

    from llmling_agent_config.session import SessionQuery


def build_message_filters(
    query: AsyncSelectRequestBuilder, filters: SessionQuery
) -> AsyncSelectRequestBuilder:
    """Apply filters to a PostgREST query."""
    if filters.name:
        query = query.eq("conversation_id", filters.name)
    if filters.agents:
        query = query.in_("name", list(filters.agents))
    if filters.since:
        query = query.gte("timestamp", filters.since)
    if filters.until:
        query = query.lte("timestamp", filters.until)
    if filters.contains:
        query = query.ilike("content", f"%{filters.contains}%")
    if filters.roles:
        query = query.in_("role", list(filters.roles))
    return query


def to_chat_message(row: dict[str, Any]) -> ChatMessage[str]:
    """Convert database row to ChatMessage."""
    cost_info = None
    if row.get("total_tokens") is not None:
        cost_info = TokenCost(
            token_usage={
                "total": row["total_tokens"] or 0,
                "prompt": row["prompt_tokens"] or 0,
                "completion": row["completion_tokens"] or 0,
            },
            total_cost=row["cost"] or 0.0,
        )

    return ChatMessage[str](
        content=row["content"],
        role=row["role"],
        name=row["name"],
        model=row["model"],
        cost_info=cost_info,
        response_time=row["response_time"],
        forwarded_from=row["forwarded_from"] or [],
        timestamp=row["timestamp"],
    )


def count_query(query: AsyncRequestBuilder) -> AsyncSelectRequestBuilder:
    """Create count query from filter builder."""
    return query.select("id", count=CountMethod.exact)
