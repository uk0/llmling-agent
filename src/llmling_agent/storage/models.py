"""Logging configuration for llmling_agent."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from pydantic import BaseModel, ConfigDict
from sqlalchemy import Column, DateTime
from sqlmodel import JSON, Field, Session, SQLModel, select
from sqlmodel.main import SQLModelConfig


if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage

    from llmling_agent.common_types import MessageRole
    from llmling_agent.models.agents import ToolCallInfo
    from llmling_agent.models.messages import TokenAndCostResult


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

    timestamp: datetime = Field(
        sa_column=Column(DateTime, default=datetime.now), default_factory=datetime.now
    )
    """When the command was executed"""

    model_config = SQLModelConfig(use_attribute_docstrings=True)

    @classmethod
    def log(
        cls,
        *,
        agent_name: str,
        session_id: str,
        command: str,
    ):
        from llmling_agent.storage import engine

        with Session(engine) as session:
            history = cls(session_id=session_id, agent_name=agent_name, command=command)
            session.add(history)
            session.commit()

    @classmethod
    def get_commands(
        cls,
        agent_name: str,
        session_id: str,
        *,
        limit: int | None = None,
        current_session_only: bool = False,
    ) -> list[str]:
        """Get command history ordered by newest first."""
        from sqlalchemy import desc

        from llmling_agent.storage import engine

        with Session(engine) as session:
            query = select(cls)
            if current_session_only:
                query = query.where(cls.session_id == str(session_id))
            else:
                query = query.where(cls.agent_name == agent_name)

            # Use the column reference from the model class
            query = query.order_by(desc(cls.timestamp))  # type: ignore
            if limit:
                query = query.limit(limit)
            return [h.command for h in session.exec(query)]


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

    model_config = SQLModelConfig(use_attribute_docstrings=True)

    @staticmethod
    def _parse_model_info(model: str | None) -> tuple[str | None, str | None]:
        """Parse model string into provider and name.

        Args:
            model: Full model string (e.g., "openai:gpt-4", "anthropic/claude-2")

        Returns:
            Tuple of (provider, name)
        """
        if not model:
            return None, None

        # Try splitting by ':' or '/'
        parts = model.split(":") if ":" in model else model.split("/")

        if len(parts) == 2:  # noqa: PLR2004
            provider, name = parts
            return provider.lower(), name

        # No provider specified, try to infer
        name = parts[0]
        if name.startswith(("gpt-", "text-", "dall-e")):
            return "openai", name
        if name.startswith("claude"):
            return "anthropic", name
        if name.startswith(("llama", "mistral")):
            return "meta", name
        # Add more provider inference rules as needed

        return None, name

    @classmethod
    def log(
        cls,
        *,
        conversation_id: str,
        content: str,
        role: MessageRole,
        name: str | None = None,
        cost_info: TokenAndCostResult | None = None,
        model: str | None = None,
        response_time: float | None = None,
        forwarded_from: list[str] | None = None,
    ):
        """Log a message with complete information."""
        from llmling_agent.storage import engine

        provider, model_name = cls._parse_model_info(model)

        with Session(engine) as session:
            msg = cls(
                conversation_id=conversation_id,
                role=role,
                name=name,
                content=content,
                model=model,  # Keep original for backwards compatibility
                model_provider=provider,
                model_name=model_name,
                response_time=response_time,
                total_tokens=cost_info.token_usage["total"] if cost_info else None,
                prompt_tokens=cost_info.token_usage["prompt"] if cost_info else None,
                completion_tokens=cost_info.token_usage["completion"]
                if cost_info
                else None,
                cost=cost_info.total_cost if cost_info else None,
                forwarded_from=forwarded_from,
            )
            session.add(msg)
            session.commit()

    @classmethod
    def to_pydantic_ai_messages(
        cls,
        conversation_id: str,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        roles: set[MessageRole] | None = None,
        limit: int | None = None,
    ) -> list[ModelMessage]:
        """Convert database messages to pydantic-ai messages.

        Args:
            conversation_id: ID of conversation to load
            since: Only include messages after this time
            until: Only include messages before this time
            roles: Only include messages with these roles
            limit: Maximum number of messages to return

        Returns:
            List of pydantic-ai ModelMessages in chronological order
        """
        from pydantic_ai.messages import (
            ModelRequest,
            ModelResponse,
            SystemPromptPart,
            TextPart,
            UserPromptPart,
        )

        from llmling_agent.storage import engine

        query = (
            select(cls)
            .where(cls.conversation_id == conversation_id)
            .order_by(cls.timestamp)  # type: ignore
        )

        if since:
            query = query.where(cls.timestamp >= since)
        if until:
            query = query.where(cls.timestamp <= until)
        if roles:
            query = query.where(cls.role.in_(roles))  # type: ignore
        if limit:
            query = query.limit(limit)

        with Session(engine) as session:
            messages = session.exec(query).all()

            result: list[ModelMessage] = []
            for msg in messages:
                match msg.role:
                    case "user":
                        result.append(
                            ModelRequest(parts=[UserPromptPart(content=msg.content)])
                        )
                    case "assistant":
                        result.append(
                            ModelResponse(parts=[TextPart(content=msg.content)])
                        )
                    case "system":
                        result.append(
                            ModelRequest(parts=[SystemPromptPart(content=msg.content)])
                        )

            return result


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

    model_config = SQLModelConfig(use_attribute_docstrings=True)

    @classmethod
    def log(
        cls,
        *,
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
    """Unique identifier for the conversation"""

    agent_name: str = Field(index=True)
    """Name of the agent handling the conversation"""

    start_time: datetime = Field(
        default_factory=datetime.now, sa_column=Column(DateTime, index=True)
    )
    """When the conversation started"""

    total_tokens: int = Field(default=0)
    """Total number of tokens used in this conversation"""

    total_cost: float = Field(default=0.0)
    """Total cost of this conversation in USD"""

    model_config = SQLModelConfig(use_attribute_docstrings=True)

    @classmethod
    def log(
        cls,
        *,
        conversation_id: str,
        name: str,
    ):
        from llmling_agent.storage import engine

        with Session(engine) as session:
            convo = cls(id=conversation_id, agent_name=name)
            session.add(convo)
            session.commit()
