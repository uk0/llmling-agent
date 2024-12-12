from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Literal

from pydantic_ai import RunContext  # noqa: TC002

from llmling_agent.context import AgentContext  # noqa: TC001
from llmling_agent.history import (
    StatsFilters,
    format_stats,
    get_conversation_stats,
    get_filtered_conversations,
)
from llmling_agent.history.formatters import format_output


if TYPE_CHECKING:
    from llmling_agent.config.capabilities import Capabilities


class HistoryTools:
    """History-related tools for LLM agents."""

    def __init__(self, agent_context: AgentContext) -> None:
        self._context = agent_context

    @property
    def capabilities(self) -> Capabilities:
        """Get agent's capabilities."""
        return self._context.capabilities

    async def search_history(  # noqa: D417
        self,
        ctx: RunContext[AgentContext],
        query: str | None = None,
        hours: int = 24,
        limit: int = 5,
    ) -> str:
        """Search conversation history.

        Search through past conversations and their messages.

        Args:
            query: Text to search for in messages
            hours: Show conversations from last N hours (default: 24)
            limit: Maximum number of conversations to show (default: 5)

        Examples:
            - "search_history('database schema')"
            - "search_history(hours=1)"  # last hour only
            - "search_history(None, hours=48, limit=10)"  # more results
        """
        if self.capabilities.history_access == "none":
            msg = "No permission to access history"
            raise PermissionError(msg)

        results = get_filtered_conversations(
            query=query,
            period=f"{hours}h",
            limit=limit,
            include_tokens=True,
        )
        return format_output(results, output_format="text")

    async def show_statistics(  # noqa: D417
        self,
        ctx: RunContext[AgentContext],
        group_by: Literal["agent", "model", "hour", "day"] = "model",
        hours: int = 24,
    ) -> str:
        """Show usage statistics for conversations.

        View statistics about conversations grouped by different criteria.

        Args:
            group_by: How to group the statistics:
                     - agent: By agent name
                     - model: By model used
                     - hour: By hour
                     - day: By day
            hours: Include stats from last N hours (default: 24)

        Examples:
            - "show_statistics('model')"  # model usage
            - "show_statistics('hour', hours=1)"  # hourly breakdown
            - "show_statistics('agent', hours=168)"  # weekly by agent
        """
        if self.capabilities.stats_access == "none":
            msg = "No permission to view statistics"
            raise PermissionError(msg)

        cutoff = datetime.now() - timedelta(hours=hours)
        stats = get_conversation_stats(
            StatsFilters(
                cutoff=cutoff,
                group_by=group_by,
            )
        )
        formatted = format_stats(stats, f"{hours}h", group_by)
        return format_output(formatted, output_format="text")
