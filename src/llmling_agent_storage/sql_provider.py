"""SQLModel-based storage provider."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import JSON, Column, Engine, and_, or_
from sqlalchemy.sql import expression
from sqlmodel import Session, select

from llmling_agent.log import get_logger
from llmling_agent.models.messages import ChatMessage, TokenCost
from llmling_agent_storage.base import StorageProvider


if TYPE_CHECKING:
    from sqlmodel.sql.expression import SelectOfScalar

    from llmling_agent.models.agents import ToolCallInfo
    from llmling_agent.models.session import SessionQuery


logger = get_logger(__name__)


class SQLModelProvider(StorageProvider):
    """Storage provider using SQLModel.

    Can work with any database supported by SQLAlchemy/SQLModel.
    Provides efficient SQL-based filtering and storage.
    """

    can_load_history = True

    def __init__(self, engine: Engine):
        """Initialize provider with database engine.

        Args:
            engine: SQLModel engine instance
        """
        self.engine = engine

    async def initialize(self) -> None:
        """Create database tables."""
        from llmling_agent.storage.models import SQLModel

        SQLModel.metadata.create_all(self.engine)

    async def cleanup(self) -> None:
        """Clean up database resources."""
        self.engine.dispose()

    async def filter_messages(
        self,
        query: SessionQuery,
    ) -> list[ChatMessage[str]]:
        """Filter messages using SQL queries."""
        with Session(self.engine) as session:
            stmt = self._build_message_query(query)
            messages = session.exec(stmt).all()
            return [self._to_chat_message(msg) for msg in messages]

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
        """Log message to database."""
        from llmling_agent.storage.models import Message

        provider, model_name = self._parse_model_info(model)

        with Session(self.engine) as session:
            msg = Message(
                conversation_id=conversation_id,
                content=content,
                role=role,
                name=name,
                model=model,
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

    async def log_conversation(
        self,
        *,
        conversation_id: str,
        agent_name: str,
        start_time: datetime | None = None,
    ) -> None:
        """Log conversation to database."""
        from llmling_agent.storage.models import Conversation

        with Session(self.engine) as session:
            conversation = Conversation(
                id=conversation_id,
                agent_name=agent_name,
                start_time=start_time or datetime.now(),
            )
            session.add(conversation)
            session.commit()

    async def log_tool_call(
        self,
        *,
        conversation_id: str,
        message_id: str,
        tool_call: ToolCallInfo,
    ) -> None:
        """Log tool call to database."""
        from llmling_agent.storage.models import ToolCall

        with Session(self.engine) as session:
            call = ToolCall(
                conversation_id=conversation_id,
                message_id=message_id,
                tool_call_id=tool_call.tool_call_id,
                timestamp=tool_call.timestamp,
                tool_name=tool_call.tool_name,
                args=tool_call.args,
                result=str(tool_call.result),
            )
            session.add(call)
            session.commit()

    async def log_command(
        self,
        *,
        agent_name: str,
        session_id: str,
        command: str,
    ) -> None:
        """Log command to database."""
        from llmling_agent.storage.models import CommandHistory

        with Session(self.engine) as session:
            history = CommandHistory(
                session_id=session_id,
                agent_name=agent_name,
                command=command,
            )
            session.add(history)
            session.commit()

    async def get_commands(
        self,
        agent_name: str,
        session_id: str,
        *,
        limit: int | None = None,
        current_session_only: bool = False,
    ) -> list[str]:
        """Get command history from database."""
        from sqlalchemy import desc

        from llmling_agent.storage.models import CommandHistory

        with Session(self.engine) as session:
            query = select(CommandHistory)
            if current_session_only:
                query = query.where(CommandHistory.session_id == str(session_id))
            else:
                query = query.where(CommandHistory.agent_name == agent_name)

            query = query.order_by(desc(CommandHistory.timestamp))  # type: ignore
            if limit:
                query = query.limit(limit)

            return [h.command for h in session.exec(query)]

    def _build_message_query(
        self,
        query: SessionQuery,
    ) -> SelectOfScalar:
        """Build SQLModel query from SessionQuery."""
        from llmling_agent.storage.models import Message

        stmt = select(Message).order_by(Message.timestamp)  # type: ignore

        conditions = []
        if query.name:
            conditions.append(Message.conversation_id == query.name)
        if query.agents:
            agent_conditions = [Column("name").in_(query.agents)]
            if query.include_forwarded:
                agent_conditions.append(
                    and_(
                        Column("forwarded_from").isnot(None),
                        expression.cast(Column("forwarded_from"), JSON).contains(
                            list(query.agents)
                        ),  # type: ignore
                    )
                )
            conditions.append(or_(*agent_conditions))
        if query.since and (cutoff := query.get_time_cutoff()):
            conditions.append(Message.timestamp >= cutoff)
        if query.until:
            conditions.append(Message.timestamp <= datetime.fromisoformat(query.until))
        if query.contains:
            conditions.append(Message.content.contains(query.contains))  # type: ignore
        if query.roles:
            conditions.append(Message.role.in_(query.roles))  # type: ignore

        if conditions:
            stmt = stmt.where(and_(*conditions))
        if query.limit:
            stmt = stmt.limit(query.limit)

        return stmt  # type: ignore

    def _to_chat_message(self, db_message: Any) -> ChatMessage[str]:
        """Convert database message to ChatMessage."""
        if db_message.cost_info:
            cost_info = TokenCost(
                token_usage={
                    "total": db_message.total_tokens or 0,
                    "prompt": db_message.prompt_tokens or 0,
                    "completion": db_message.completion_tokens or 0,
                },
                total_cost=db_message.cost or 0.0,
            )
        else:
            cost_info = None

        return ChatMessage[str](
            content=db_message.content,
            role=db_message.role,
            name=db_message.name,
            model=db_message.model,
            cost_info=cost_info,
            response_time=db_message.response_time,
            forwarded_from=db_message.forwarded_from or [],
        )

    @staticmethod
    def _parse_model_info(model: str | None) -> tuple[str | None, str | None]:
        """Parse model string into provider and name."""
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

        return None, name
