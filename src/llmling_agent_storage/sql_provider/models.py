"""Logging configuration for llmling_agent."""

from __future__ import annotations

from datetime import UTC, datetime
import importlib.util
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict
from sqlalchemy import Column, DateTime
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.types import TypeDecorator
from sqlmodel import JSON, Field, SQLModel
from sqlmodel.main import SQLModelConfig

from llmling_agent.utils.now import get_now


REFLEX_INSTALLED = importlib.util.find_spec("reflex") is not None


class UTCDateTime(TypeDecorator):
    """Stores DateTime as UTC."""

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect):
        if value is not None:
            if value.tzinfo is None:
                value = value.replace(tzinfo=UTC)
            else:
                value = value.astimezone(UTC)
        return value

    def process_result_value(self, value: datetime | None, dialect):
        if value is not None:
            value = value.replace(tzinfo=UTC)
        return value


class CommandHistory(AsyncAttrs, SQLModel, table=True):  # type: ignore[call-arg]
    """Database model for command history."""

    id: int = Field(default=None, primary_key=True)
    """Primary key for command history entry"""

    session_id: str = Field(index=True)
    """ID of the chat session"""

    agent_name: str = Field(index=True)
    """Name of the agent that executed the command"""

    command: str
    """The command that was executed"""

    context_type: str | None = Field(default=None, index=True)
    """Type of the command context (e.g. 'AgentContext', 'PoolSupervisor', etc.)"""

    context_metadata: dict[str, int | str | bool | float] = Field(
        default_factory=dict, sa_column=Column(JSON)
    )
    """Additional context information about command execution"""

    timestamp: datetime = Field(
        sa_column=Column(UTCDateTime, default=get_now), default_factory=get_now
    )
    """When the command was executed"""

    if not REFLEX_INSTALLED:
        model_config = SQLModelConfig(use_attribute_docstrings=True)  # pyright: ignore[reportCallIssue]


class MessageLog(BaseModel):
    """Raw message log entry."""

    timestamp: datetime
    """When the message was sent"""

    role: str
    """Role of the message sender (user/assistant/system)"""

    content: str
    """Content of the message"""

    token_usage: dict[str, int] | None = None
    """Token usage statistics as provided by model"""

    cost: float | None = None
    """Cost of generating this message in USD"""

    model: str | None = None
    """Name of the model that generated this message"""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True, extra="forbid")


class ConversationLog(BaseModel):
    """Collection of messages forming a conversation."""

    id: str
    """Unique identifier for the conversation"""

    agent_name: str
    """Name of the agent handling the conversation"""

    start_time: datetime
    """When the conversation started"""

    messages: list[MessageLog]
    """List of messages in the conversation"""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True, extra="forbid")


class Message(AsyncAttrs, SQLModel, table=True):  # type: ignore[call-arg]
    """Database model for message logs."""

    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    """Unique identifier for the message"""

    conversation_id: str = Field(index=True)
    """ID of the conversation this message belongs to"""

    timestamp: datetime = Field(
        sa_column=Column(UTCDateTime, default=get_now), default_factory=get_now
    )
    """When the message was sent"""

    role: str
    """Role of the message sender (user/assistant/system)"""

    name: str | None = Field(default=None, index=True)
    """Display name of the sender"""

    content: str
    """Content of the message"""

    model: str | None = None
    """Full model identifier (including provider)"""

    model_name: str | None = Field(default=None, index=True)
    """Name of the model (e.g., "gpt-4")"""

    model_provider: str | None = Field(default=None, index=True)
    """Provider of the model (e.g., "openai")"""

    forwarded_from: list[str] | None = Field(default=None, sa_column=Column(JSON))
    """List of agent names that forwarded this message"""

    total_tokens: int | None = Field(default=None, index=True)
    """Total number of tokens used"""

    prompt_tokens: int | None = None
    """Number of tokens in the prompt"""

    completion_tokens: int | None = None
    """Number of tokens in the completion"""

    cost: float | None = Field(default=None, index=True)
    """Cost of generating this message in USD"""

    response_time: float | None = None
    """Time taken to generate the response in seconds"""

    checkpoint_data: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))
    """A dictionary of checkpoints (name -> metadata)."""

    if not REFLEX_INSTALLED:
        model_config = SQLModelConfig(use_attribute_docstrings=True)  # pyright: ignore[reportCallIssue]


class ToolCall(AsyncAttrs, SQLModel, table=True):  # type: ignore[call-arg]
    """Record of a tool being called during a conversation."""

    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    """Unique identifier for the tool call"""

    conversation_id: str = Field(index=True)
    """ID of the conversation where the tool was called"""

    message_id: str = Field(index=True)
    """ID of the message that triggered this tool call"""

    timestamp: datetime = Field(
        sa_column=Column(UTCDateTime, default=get_now), default_factory=get_now
    )
    """When the tool was called"""

    tool_call_id: str | None = None
    """ID provided by the LLM for this tool call"""

    tool_name: str
    """Name of the tool that was called"""

    args: dict = Field(sa_column=Column(JSON))
    """Arguments passed to the tool"""

    result: str = Field(...)
    """Result returned by the tool"""

    if not REFLEX_INSTALLED:
        model_config = SQLModelConfig(use_attribute_docstrings=True)  # pyright: ignore[reportCallIssue]


class Conversation(AsyncAttrs, SQLModel, table=True):  # type: ignore[call-arg]
    """Database model for conversations."""

    id: str = Field(primary_key=True)
    """Unique identifier for the conversation"""

    agent_name: str = Field(index=True)
    """Name of the agent handling the conversation"""

    start_time: datetime = Field(
        sa_column=Column(UTCDateTime, index=True), default_factory=get_now
    )
    """When the conversation started"""

    total_tokens: int = 0
    """Total number of tokens used in this conversation"""

    total_cost: float = 0.0
    """Total cost of this conversation in USD"""

    if not REFLEX_INSTALLED:
        model_config = SQLModelConfig(use_attribute_docstrings=True)  # pyright: ignore[reportCallIssue]
