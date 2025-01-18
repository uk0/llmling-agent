"""Logging configuration for llmling_agent."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict
from sqlalchemy import Column, DateTime
from sqlmodel import JSON, Field, SQLModel
from sqlmodel.main import SQLModelConfig


if TYPE_CHECKING:
    from llmling_agent.common_types import JsonValue


class CommandHistory(SQLModel, table=True):  # type: ignore[call-arg]
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

    context_metadata: dict[str, JsonValue] = Field(
        default_factory=dict, sa_column=Column(JSON)
    )
    """Additional context information about command execution"""

    timestamp: datetime = Field(
        sa_column=Column(DateTime, default=datetime.now), default_factory=datetime.now
    )
    """When the command was executed"""

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

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)


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

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)


class Message(SQLModel, table=True):  # type: ignore[call-arg]
    """Database model for message logs."""

    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    """Unique identifier for the message"""

    conversation_id: str = Field(index=True)
    """ID of the conversation this message belongs to"""

    timestamp: datetime = Field(sa_column=Column(DateTime), default_factory=datetime.now)
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

    model_config = SQLModelConfig(use_attribute_docstrings=True)  # pyright: ignore[reportCallIssue]


class ToolCall(SQLModel, table=True):  # type: ignore[call-arg]
    """Record of a tool being called during a conversation."""

    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    """Unique identifier for the tool call"""

    conversation_id: str = Field(index=True)
    """ID of the conversation where the tool was called"""

    message_id: str = Field(index=True)
    """ID of the message that triggered this tool call"""

    timestamp: datetime = Field(sa_column=Column(DateTime), default_factory=datetime.now)
    """When the tool was called"""

    tool_call_id: str | None = None
    """ID provided by the LLM for this tool call"""

    tool_name: str
    """Name of the tool that was called"""

    args: dict = Field(sa_column=Column(JSON))
    """Arguments passed to the tool"""

    result: str = Field(...)
    """Result returned by the tool"""

    model_config = SQLModelConfig(use_attribute_docstrings=True)  # pyright: ignore[reportCallIssue]


class Conversation(SQLModel, table=True):  # type: ignore[call-arg]
    """Database model for conversations."""

    id: str = Field(primary_key=True)
    """Unique identifier for the conversation"""

    agent_name: str = Field(index=True)
    """Name of the agent handling the conversation"""

    start_time: datetime = Field(
        default_factory=datetime.now, sa_column=Column(DateTime, index=True)
    )
    """When the conversation started"""

    total_tokens: int = 0
    """Total number of tokens used in this conversation"""

    total_cost: float = 0.0
    """Total cost of this conversation in USD"""

    model_config = SQLModelConfig(use_attribute_docstrings=True)  # pyright: ignore[reportCallIssue]
