"""Supabase storage provider implementation."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Self

from supabase import AsyncClient, create_async_client

from llmling_agent.log import get_logger
from llmling_agent.messaging.messages import TokenCost
from llmling_agent.utils.now import get_now
from llmling_agent_storage.base import StorageProvider
from llmling_agent_storage.models import ConversationData, QueryFilters, StatsFilters
from llmling_agent_storage.supabase_provider.queries import (
    CREATE_COMMANDS_TABLE,
    CREATE_CONVERSATIONS_TABLE,
    CREATE_MESSAGES_TABLE,
    CREATE_TOOL_CALLS_TABLE,
)
from llmling_agent_storage.supabase_provider.utils import (
    build_message_filters,
    to_chat_message,
)


if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

    from llmling_agent.common_types import JsonValue
    from llmling_agent.messaging.messages import ChatMessage
    from llmling_agent.tools import ToolCallInfo
    from llmling_agent_config.session import SessionQuery
    from llmling_agent_config.storage import SupabaseConfig

logger = get_logger(__name__)


class SupabaseProvider(StorageProvider):
    """Supabase storage provider using async client."""

    can_load_history = True

    def __init__(self, config: SupabaseConfig):
        """Initialize async Supabase client."""
        super().__init__(config)
        self.config: SupabaseConfig = config
        self.client: AsyncClient | None = None

    async def __aenter__(self) -> Self:
        """Initialize async client and tables."""
        url = self.config.supabase_url or os.getenv("SUPABASE_PROJECT_URL")
        api_key = self.config.key.get_secret_value() or os.getenv("SUPABASE_API_KEY")
        assert url, "Supabase URL not provided"
        assert api_key, "Supabase API key not provided"
        self.client = await create_async_client(url, api_key)
        await self._init_tables()
        return self

    async def _init_tables(self):
        """Initialize database tables."""
        if not self.client:
            msg = "Client not initialized"
            raise RuntimeError(msg)

        queries = [
            CREATE_MESSAGES_TABLE,
            CREATE_CONVERSATIONS_TABLE,
            CREATE_TOOL_CALLS_TABLE,
            CREATE_COMMANDS_TABLE,
        ]
        for query in queries:
            await self.client.rpc("exec_query", {"query": query}).execute()

    async def filter_messages(self, query: SessionQuery) -> list[ChatMessage[str]]:
        """Get messages using async PostgREST client."""
        assert self.client, "Client not initialized"
        q = self.client.from_("messages").select("*")
        q = build_message_filters(q, query)

        if query.limit:
            q = q.limit(query.limit)

        result = await q.execute()
        return [to_chat_message(r) for r in result.data]

    async def log_message(
        self,
        *,
        conversation_id: str,
        message_id: str,
        content: str,
        role: str,
        name: str | None = None,
        cost_info: TokenCost | None = None,
        model: str | None = None,
        response_time: float | None = None,
        forwarded_from: list[str] | None = None,
    ):
        """Log message using async client."""
        assert self.client, "Client not initialized"
        provider, model_name = None, None
        if model and ":" in model:
            provider, model_name = model.split(":", 1)
        else:
            model_name = model

        data = {
            "conversation_id": conversation_id,
            "id": message_id,
            "content": content,
            "role": role,
            "name": name,
            "model": model,
            "model_provider": provider,
            "model_name": model_name,
            "response_time": response_time,
            "forwarded_from": forwarded_from,
            "total_tokens": cost_info.token_usage.get("total") if cost_info else None,
            "prompt_tokens": cost_info.token_usage.get("prompt") if cost_info else None,
            "completion_tokens": cost_info.token_usage.get("completion")
            if cost_info
            else None,
            "cost": float(cost_info.total_cost) if cost_info else None,
        }
        await self.client.from_("messages").insert(data).execute()

    async def log_conversation(
        self,
        *,
        conversation_id: str,
        node_name: str,
        start_time: datetime | None = None,
    ):
        """Log conversation start."""
        assert self.client, "Client not initialized"
        data = {
            "id": conversation_id,
            "agent_name": node_name,
            "start_time": start_time or get_now(),
        }
        await self.client.from_("conversations").insert(data).execute()

    async def log_tool_call(
        self,
        *,
        conversation_id: str,
        message_id: str,
        tool_call: ToolCallInfo,
    ):
        """Log tool call."""
        assert self.client, "Client not initialized"
        data = {
            "conversation_id": conversation_id,
            "message_id": message_id,
            "tool_call_id": tool_call.tool_call_id,
            "timestamp": tool_call.timestamp,
            "tool_name": tool_call.tool_name,
            "args": tool_call.args,
            "result": str(tool_call.result),
        }
        await self.client.from_("tool_calls").insert(data).execute()

    async def log_command(
        self,
        *,
        agent_name: str,
        session_id: str,
        command: str,
        context_type: type | None = None,
        metadata: dict[str, JsonValue] | None = None,
    ):
        """Log command execution."""
        assert self.client, "Client not initialized"
        data = {
            "session_id": session_id,
            "agent_name": agent_name,
            "command": command,
            "context_type": context_type.__name__ if context_type else None,
            "metadata": metadata or {},
        }
        await self.client.from_("commands").insert(data).execute()

    async def get_commands(
        self,
        agent_name: str,
        session_id: str,
        *,
        limit: int | None = None,
        current_session_only: bool = False,
    ) -> list[str]:
        """Get command history."""
        assert self.client, "Client not initialized"
        q = self.client.from_("commands").select("command")

        if current_session_only:
            q = q.eq("session_id", session_id)
        else:
            q = q.eq("agent_name", agent_name)

        q = q.order("timestamp", desc=True)
        if limit:
            q = q.limit(limit)

        result = await q.execute()
        return [r["command"] for r in result.data]

    async def get_conversations(
        self,
        filters: QueryFilters,
    ) -> list[tuple[ConversationData, Sequence[ChatMessage[str]]]]:
        """Get filtered conversations with messages."""
        assert self.client, "Client not initialized"
        results: list[tuple[ConversationData, Sequence[ChatMessage[str]]]] = []

        # Query conversations
        q = self.client.from_("conversations").select("*")
        if filters.agent_name:
            q = q.eq("agent_name", filters.agent_name)
        if filters.since:
            q = q.gte("start_time", filters.since)

        q = q.order("start_time", desc=True)
        if filters.limit:
            q = q.limit(filters.limit)

        conversations = await q.execute()

        for conv in conversations.data:
            # Get messages for each conversation
            msg_q = (
                self.client.from_("messages")
                .select("*")
                .eq("conversation_id", conv["id"])
            )
            if filters.query:
                msg_q = msg_q.ilike("content", f"%{filters.query}%")
            if filters.model:
                msg_q = msg_q.eq("model", filters.model)

            messages = await msg_q.execute()

            if not messages.data:
                continue

            chat_messages = [to_chat_message(msg) for msg in messages.data]

            # Calculate token usage
            token_usage = None
            if any(msg.cost_info for msg in chat_messages):
                total = prompt = completion = 0
                for msg in chat_messages:
                    if msg.cost_info:
                        total += msg.cost_info.token_usage["total"]
                        prompt += msg.cost_info.token_usage["prompt"]
                        completion += msg.cost_info.token_usage["completion"]
                token_usage = {"total": total, "prompt": prompt, "completion": completion}

            conv_data = ConversationData(
                id=conv["id"],
                agent=conv["agent_name"],
                start_time=conv["start_time"].isoformat(),
                messages=[
                    {
                        "role": msg.role,
                        "content": msg.content,
                        "timestamp": msg.timestamp.isoformat(),
                        "model": msg.model,
                        "name": msg.name,
                        "token_usage": msg.cost_info.token_usage
                        if msg.cost_info
                        else None,
                        "cost": msg.cost_info.total_cost if msg.cost_info else None,
                        "response_time": msg.response_time,
                    }
                    for msg in chat_messages
                ],
                token_usage=token_usage,  # type: ignore
            )
            results.append((conv_data, chat_messages))

        return results

    async def get_conversation_stats(
        self,
        filters: StatsFilters,
    ) -> dict[str, dict[str, Any]]:
        """Get conversation statistics."""
        # Get raw data
        assert self.client, "Client not initialized"
        q = (
            self.client.from_("messages")
            .select(
                "model,model_name,name,timestamp,total_tokens,prompt_tokens,completion_tokens"
            )
            .gte("timestamp", filters.cutoff)
        )

        if filters.agent_name:
            q = q.eq("name", filters.agent_name)

        result = await q.execute()

        # Convert to format expected by aggregate_stats
        rows = [
            (
                row["model"],
                row["name"],
                row["timestamp"],
                TokenCost(
                    token_usage={
                        "total": row["total_tokens"] or 0,
                        "prompt": row["prompt_tokens"] or 0,
                        "completion": row["completion_tokens"] or 0,
                    },
                    total_cost=0.0,
                )
                if row["total_tokens"]
                else None,
            )
            for row in result.data
        ]

        return self.aggregate_stats(rows, filters.group_by)

    async def cleanup(self):
        """Close client connection."""
        if self.client:
            # await self.client.aclose()
            self.client = None
