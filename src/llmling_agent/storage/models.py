"""Logging configuration for llmling_agent."""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from pydantic import BaseModel, ConfigDict
from sqlalchemy import Column, DateTime
from sqlmodel import JSON, Field, SQLModel

from llmling_agent.pydantic_ai_utils import TokenUsage  # noqa: TC001


class CommandHistory(SQLModel, table=True):  # type: ignore[call-arg]
    """Database model for command history."""

    id: int = Field(default=None, primary_key=True)
    session_id: str = Field(index=True)
    agent_name: str = Field(index=True)
    command: str
    timestamp: datetime = Field(
        sa_column=Column(DateTime, default=datetime.now), default_factory=datetime.now
    )


class MessageLog(BaseModel):
    """Raw message log entry."""

    timestamp: datetime
    role: str
    content: str
    token_usage: dict[str, int] | None = None  # as provided by model
    cost: float | None = None
    model: str | None = None

    model_config = ConfigDict(frozen=True)


class ConversationLog(BaseModel):
    """Collection of messages forming a conversation."""

    id: str
    agent_name: str
    start_time: datetime
    messages: list[MessageLog]

    model_config = ConfigDict(frozen=True)


class Message(SQLModel, table=True):  # type: ignore[call-arg]
    """Database model for message logs."""

    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    conversation_id: str = Field(index=True)
    timestamp: datetime = Field(sa_column=Column(DateTime), default_factory=datetime.now)
    role: str
    content: str
    token_usage: TokenUsage | None = Field(None, sa_column=Column(JSON))
    cost: float | None = Field(default=None)
    model: str | None = Field(default=None)


class Conversation(SQLModel, table=True):  # type: ignore[call-arg]
    """Database model for conversations."""

    id: str = Field(primary_key=True)
    agent_name: str = Field(index=True)
    start_time: datetime = Field(
        default_factory=datetime.now, sa_column=Column(DateTime, index=True)
    )
    total_tokens: int = Field(default=0)
    total_cost: float = Field(default=0.0)
