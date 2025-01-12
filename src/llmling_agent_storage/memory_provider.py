from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from llmling_agent_storage.base import StorageProvider


if TYPE_CHECKING:
    from llmling_agent.models.agents import ToolCallInfo
    from llmling_agent.models.messages import ChatMessage, TokenCost
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
