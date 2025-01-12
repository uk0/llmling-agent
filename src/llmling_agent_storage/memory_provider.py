from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from llmling_agent.history.models import ConversationData, QueryFilters, StatsFilters
from llmling_agent.models.messages import ChatMessage, TokenCost
from llmling_agent_storage.base import StorageProvider


if TYPE_CHECKING:
    from collections.abc import Sequence

    from tokonomics.toko_types import TokenUsage

    from llmling_agent.models.agents import ToolCallInfo
    from llmling_agent.models.session import SessionQuery


class MemoryStorageProvider(StorageProvider):
    """In-memory storage provider for testing."""

    can_load_history = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.messages: list[dict] = []
        self.conversations: list[dict] = []
        self.tool_calls: list[dict] = []
        self.commands: list[dict] = []

    def cleanup(self):
        """Clear all stored data."""
        self.messages.clear()
        self.conversations.clear()
        self.tool_calls.clear()
        self.commands.clear()

    async def filter_messages(
        self,
        query: SessionQuery,
    ) -> list[ChatMessage[str]]:
        """Filter messages from memory."""
        from llmling_agent.models.messages import ChatMessage

        filtered = []
        for msg in self.messages:
            if query.name and msg["conversation_id"] != query.name:
                continue
            # ... apply other filters ...
            filtered.append(
                ChatMessage(
                    content=msg["content"],
                    role=msg["role"],
                    name=msg["name"],
                    model=msg["model"],
                )
            )
        return filtered

    async def log_message(
        self,
        *,
        conversation_id: str,
        content: str,
        role: str,
        name: str | None = None,
        cost_info: TokenCost | None = None,
        model: str | None = None,
        response_time: float | None = None,
        forwarded_from: list[str] | None = None,
    ) -> None:
        """Store message in memory."""
        self.messages.append({
            "conversation_id": conversation_id,
            "content": content,
            "role": role,
            "name": name,
            "cost_info": cost_info.token_usage if cost_info else None,
            "model": model,
            "response_time": response_time,
            "forwarded_from": forwarded_from,
            "timestamp": datetime.now(),
        })

    async def log_conversation(
        self,
        *,
        conversation_id: str,
        agent_name: str,
        start_time: datetime | None = None,
    ) -> None:
        """Store conversation in memory."""
        self.conversations.append({
            "id": conversation_id,
            "agent_name": agent_name,
            "start_time": start_time or datetime.now(),
        })

    async def log_tool_call(
        self,
        *,
        conversation_id: str,
        message_id: str,
        tool_call: ToolCallInfo,
    ) -> None:
        """Store tool call in memory."""
        self.tool_calls.append({
            "conversation_id": conversation_id,
            "message_id": message_id,
            "tool_name": tool_call.tool_name,
            "args": tool_call.args,
            "result": tool_call.result,
            "timestamp": tool_call.timestamp,
        })

    async def log_command(
        self,
        *,
        agent_name: str,
        session_id: str,
        command: str,
    ) -> None:
        """Store command in memory."""
        self.commands.append({
            "agent_name": agent_name,
            "session_id": session_id,
            "command": command,
            "timestamp": datetime.now(),
        })

    async def get_commands(
        self,
        agent_name: str,
        session_id: str,
        *,
        limit: int | None = None,
        current_session_only: bool = False,
    ) -> list[str]:
        """Get commands from memory."""
        filtered = []
        for cmd in reversed(self.commands):  # newest first
            if current_session_only and cmd["session_id"] != session_id:
                continue
            if not current_session_only and cmd["agent_name"] != agent_name:
                continue
            filtered.append(cmd["command"])
            if limit and len(filtered) >= limit:
                break
        return filtered

    async def get_conversations(
        self,
        filters: QueryFilters,
    ) -> list[tuple[ConversationData, Sequence[ChatMessage[str]]]]:
        """Get filtered conversations from memory."""
        from typing import cast

        from llmling_agent.history.models import MessageData

        results: list[tuple[ConversationData, Sequence[ChatMessage[str]]]] = []

        # First get matching conversations
        convs = {}
        for conv in self.conversations:
            if filters.agent_name and conv["agent_name"] != filters.agent_name:
                continue
            if filters.since and conv["start_time"] < filters.since:
                continue
            convs[conv["id"]] = conv

        # Then get messages for each conversation
        for conv_id, conv in convs.items():
            conv_messages: list[ChatMessage[str]] = []
            for msg in self.messages:
                if msg["conversation_id"] != conv_id:
                    continue
                if filters.query and filters.query not in msg["content"]:
                    continue
                if filters.model and msg["model"] != filters.model:
                    continue

                cost_info = None
                if msg["cost_info"]:
                    cost_info = TokenCost(
                        token_usage=msg["cost_info"],
                        total_cost=msg.get("cost", 0.0),
                    )

                chat_msg = ChatMessage[str](
                    content=msg["content"],
                    role=msg["role"],
                    name=msg["name"],
                    model=msg["model"],
                    cost_info=cost_info,
                    response_time=msg["response_time"],
                    forwarded_from=msg["forwarded_from"],
                    timestamp=msg["timestamp"],
                )
                conv_messages.append(chat_msg)

            # Skip if no matching messages for content filter
            if filters.query and not conv_messages:
                continue

            # Convert ChatMessages to MessageData format for ConversationData
            message_data: list[MessageData] = [
                cast(
                    "MessageData",
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
                    },
                )
                for msg in conv_messages
            ]

            # Create conversation data with proper MessageData
            conv_data = ConversationData(
                id=conv_id,
                agent=conv["agent_name"],
                start_time=conv["start_time"].isoformat(),
                messages=message_data,  # Now using properly typed MessageData
                token_usage=self._aggregate_token_usage(conv_messages),
            )
            results.append((conv_data, conv_messages))

            if filters.limit and len(results) >= filters.limit:
                break

        return results

    async def get_conversation_stats(
        self,
        filters: StatsFilters,
    ) -> dict[str, dict[str, Any]]:
        """Get statistics from memory."""
        # Collect raw data
        rows = []
        for msg in self.messages:
            if msg["timestamp"] <= filters.cutoff:
                continue
            if filters.agent_name and msg["name"] != filters.agent_name:
                continue

            cost_info = None
            if msg["cost_info"]:
                cost_info = TokenCost(
                    token_usage=msg["cost_info"],
                    total_cost=msg.get("cost", 0.0),
                )

            rows.append((
                msg["model"],
                msg["name"],
                msg["timestamp"],
                cost_info,
            ))

        # Use base class aggregation
        return self.aggregate_stats(rows, filters.group_by)

    def _aggregate_token_usage(
        self,
        messages: Sequence[ChatMessage[Any]],
    ) -> TokenUsage:
        """Sum up tokens from a sequence of messages."""
        total = prompt = completion = 0
        for msg in messages:
            if msg.cost_info:
                total += msg.cost_info.token_usage.get("total", 0)
                prompt += msg.cost_info.token_usage.get("prompt", 0)
                completion += msg.cost_info.token_usage.get("completion", 0)
        return {"total": total, "prompt": prompt, "completion": completion}
