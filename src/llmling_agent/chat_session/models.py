"""Data models for agent chat sessions."""

from __future__ import annotations

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
