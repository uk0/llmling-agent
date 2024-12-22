"""Logging configuration for llmling_agent."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict
from sqlalchemy import Column, DateTime
from sqlmodel import JSON, Field, Session, SQLModel

from llmling_agent.pydantic_ai_utils import TokenUsage  # noqa: TC001


if TYPE_CHECKING:
    from llmling_agent.models.agents import ToolCallInfo
    from llmling_agent.models.messages import TokenAndCostResult


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

    @classmethod
    def log(
        cls,
        conversation_id: str,
        content: str,
        role: Literal["user", "assistant", "system"],
        *,
        cost_info: TokenAndCostResult | None = None,
        model: str | None = None,
    ):
        from llmling_agent.storage import engine

        with Session(engine) as session:
            msg = cls(
                conversation_id=conversation_id,
                role=role,
                content=content,
                token_usage=cost_info.token_usage if cost_info else None,
                cost=cost_info.cost_usd if cost_info else None,
                model=model,
            )
            session.add(msg)
            session.commit()


class ToolCall(SQLModel, table=True):  # type: ignore[call-arg]
    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    conversation_id: str = Field(index=True)
    message_id: str = Field(index=True)  # Link to triggering message
    timestamp: datetime = Field(sa_column=Column(DateTime), default_factory=datetime.now)
    tool_call_id: str | None = None
    tool_name: str
    args: dict = Field(sa_column=Column(JSON))
    result: str = Field(...)

    @classmethod
    def log(
        cls,
        conversation_id: str,
        message_id: str,
        tool_call: ToolCallInfo,
    ):
        """Log a tool call to the database."""
        from llmling_agent.storage import engine

        with Session(engine) as session:
            call = cls(
                conversation_id=conversation_id,
                message_id=message_id,
                tool_call_id=tool_call.tool_call_id,
                timestamp=tool_call.timestamp,
                tool_name=tool_call.tool_name,
                args=tool_call.args,
                result=str(tool_call.result),  # Convert result to string
            )
            session.add(call)
            session.commit()


class Conversation(SQLModel, table=True):  # type: ignore[call-arg]
    """Database model for conversations."""

    id: str = Field(primary_key=True)
    agent_name: str = Field(index=True)
    start_time: datetime = Field(
        default_factory=datetime.now, sa_column=Column(DateTime, index=True)
    )
    total_tokens: int = Field(default=0)
    total_cost: float = Field(default=0.0)

    @classmethod
    def log(cls, conversation_id: str, name: str):
        from llmling_agent.storage import engine

        with Session(engine) as session:
            convo = cls(id=conversation_id, agent_name=name)
            session.add(convo)
            session.commit()
