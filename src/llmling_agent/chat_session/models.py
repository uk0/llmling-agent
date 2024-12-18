"""Data models for agent chat sessions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal
from uuid import UUID  # noqa: TC003

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """A message in the conversation."""

    content: str
    role: Literal["user", "assistant", "system"]
    metadata: dict[str, Any] | None = None
    timestamp: datetime = Field(default_factory=datetime.now)


class ChatSessionMetadata(BaseModel):
    """Configuration and state of an agent chat session."""

    session_id: UUID
    agent_name: str
    model: str | None = None
    tool_states: dict[str, bool]
    start_time: datetime = Field(default_factory=datetime.now)


@dataclass
class SessionState:
    """Current state of the interactive session."""

    current_model: str | None = None
    message_count: int = 0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_cost: float = 0.0
    start_time: datetime = field(default_factory=datetime.now)
    last_command: str | None = None

    def update_tokens(self, message: ChatMessage) -> None:
        """Update token counts from message metadata."""
        if message.metadata and (token_usage := message.metadata.get("token_usage")):
            self.total_tokens += token_usage.get("total", 0)
            self.prompt_tokens += token_usage.get("prompt", 0)
            self.completion_tokens += token_usage.get("completion", 0)

        if message.metadata and (cost := message.metadata.get("cost")):
            self.total_cost += cost

    @property
    def duration(self) -> str:
        """Get formatted duration since session start."""
        duration = datetime.now() - self.start_time
        hours, remainder = divmod(int(duration.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
