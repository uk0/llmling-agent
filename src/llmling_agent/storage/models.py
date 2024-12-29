"""Logging configuration for llmling_agent."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict
from sqlalchemy import Column, DateTime
from sqlmodel import JSON, Field, Session, SQLModel, select


if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage

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
    name: str | None = Field(default=None, index=True)
    content: str
    model: str | None = Field(default=None)
    model_name: str | None = Field(default=None, index=True)  # e.g., "gpt-4"
    model_provider: str | None = Field(default=None, index=True)  # e.g., "openai"
    # Token usage
    total_tokens: int | None = Field(default=None, index=True)
    prompt_tokens: int | None = Field(default=None)
    completion_tokens: int | None = Field(default=None)

    # Cost info
    cost: float | None = Field(default=None, index=True)

    # Performance metrics
    response_time: float | None = Field(default=None)

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
        role: Literal["user", "assistant", "system"],
        name: str | None = None,
        cost_info: TokenAndCostResult | None = None,
        model: str | None = None,
        response_time: float | None = None,
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
        roles: set[Literal["user", "assistant", "system"]] | None = None,
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
    agent_name: str = Field(index=True)
    start_time: datetime = Field(
        default_factory=datetime.now, sa_column=Column(DateTime, index=True)
    )
    total_tokens: int = Field(default=0)
    total_cost: float = Field(default=0.0)

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
