"""SQLModel-based storage provider."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

from sqlalchemy import JSON, Column, Engine, and_, or_
from sqlalchemy.sql import expression
from sqlmodel import Session, desc, select

from llmling_agent.history.models import (
    ConversationData,
    MessageData,
    QueryFilters,
    StatsFilters,
)
from llmling_agent.log import get_logger
from llmling_agent.models.messages import ChatMessage, TokenCost
from llmling_agent_storage.base import StorageProvider


if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlmodel.sql.expression import SelectOfScalar
    from tokonomics.toko_types import TokenUsage

    from llmling_agent.models.agents import ToolCallInfo
    from llmling_agent.models.session import SessionQuery


logger = get_logger(__name__)


class SQLModelProvider(StorageProvider):
    """Storage provider using SQLModel.

    Can work with any database supported by SQLAlchemy/SQLModel.
    Provides efficient SQL-based filtering and storage.
    """

    can_load_history = True

    def __init__(self, engine: Engine, *, auto_migrate: bool = False, **kwargs: Any):
        """Initialize provider with database engine.

        Args:
            engine: SQLModel engine instance
            auto_migrate: Whether to automatically add missing columns
            kwargs: Additional arguments to pass to StorageProvider
        """
        from llmling_agent.storage.models import SQLModel

        super().__init__(**kwargs)
        self.engine = engine
        SQLModel.metadata.create_all(self.engine)
        self._init_database(auto_migrate=auto_migrate)

    def _init_database(self, auto_migrate: bool = True):
        """Initialize database tables and optionally migrate columns.

        Args:
            auto_migrate: Whether to automatically add missing columns
        """
        from sqlalchemy import inspect
        from sqlalchemy.sql import text

        from llmling_agent.storage.models import SQLModel

        # Create tables if they don't exist
        SQLModel.metadata.create_all(self.engine)

        # Optionally add missing columns
        if auto_migrate:
            with self.engine.connect() as conn:
                inspector = inspect(self.engine)

                # For each table in our models
                for table_name, table in SQLModel.metadata.tables.items():
                    existing = {col["name"] for col in inspector.get_columns(table_name)}

                    # For each column in model that doesn't exist in DB
                    for col in table.columns:
                        if col.name not in existing:
                            # Create ALTER TABLE statement based on column type
                            type_sql = col.type.compile(self.engine.dialect)
                            nullable = "" if col.nullable else " NOT NULL"
                            default = self._get_column_default(col)
                            sql = f"ALTER TABLE {table_name} ADD COLUMN {col.name} {type_sql}{nullable}{default}"  # noqa: E501
                            conn.execute(text(sql))

                conn.commit()

    @staticmethod
    def _get_column_default(column: Any) -> str:
        """Get SQL DEFAULT clause for column."""
        if column.default is None:
            return ""
        if hasattr(column.default, "arg"):
            # Simple default value
            return f" DEFAULT {column.default.arg}"
        if hasattr(column.default, "sqltext"):
            # Computed default
            return f" DEFAULT {column.default.sqltext}"
        return ""

    def cleanup(self):
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
    ):
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
    ):
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
    ):
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

    async def log_command(self, *, agent_name: str, session_id: str, command: str):
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

    async def get_conversations(
        self,
        filters: QueryFilters,
    ) -> list[tuple[ConversationData, Sequence[ChatMessage[str]]]]:
        """Get filtered conversations using SQL queries."""
        from sqlmodel import select

        from llmling_agent.storage.models import Conversation, Message

        with Session(self.engine) as session:
            # Explicitly type our results list
            results: list[tuple[ConversationData, Sequence[ChatMessage[str]]]] = []

            # Build conversation query
            conv_stmt = select(Conversation).order_by(desc(Conversation.start_time))
            if filters.agent_name:
                conv_stmt = conv_stmt.where(Conversation.agent_name == filters.agent_name)
            if filters.since:
                conv_stmt = conv_stmt.where(Conversation.start_time >= filters.since)
            if filters.limit:
                conv_stmt = conv_stmt.limit(filters.limit)

            for conv in session.exec(conv_stmt):
                # Get messages for this conversation
                msg_stmt = (
                    select(Message)
                    .where(Message.conversation_id == conv.id)
                    .order_by(Message.timestamp)  # type: ignore[arg-type]
                )

                if filters.query:
                    msg_stmt = msg_stmt.where(Message.content.contains(filters.query))  # type: ignore[attr-defined]
                if filters.model:
                    msg_stmt = msg_stmt.where(Message.model_name == filters.model)

                messages = session.exec(msg_stmt).all()

                # Skip conversations with no matching messages if content filtered
                if filters.query and not messages:
                    continue

                # Convert to ChatMessages
                chat_messages = [self._to_chat_message(msg) for msg in messages]

                # Convert messages to MessageData with proper typing
                message_data: list[MessageData] = [
                    cast(
                        "MessageData",
                        {
                            "role": msg.role,
                            "content": msg.content,
                            "timestamp": msg.timestamp.isoformat(),
                            "model": msg.model,
                            "name": msg.name,
                            "token_usage": msg.cost_info.token_usage
                            if msg.cost_info
                            else None,
                            "cost": msg.cost_info.total_cost if msg.cost_info else None,
                            "response_time": msg.response_time,
                        },
                    )
                    for msg in chat_messages
                ]

                # Create ConversationData
                conv_data = ConversationData(
                    id=conv.id,
                    agent=conv.agent_name,
                    start_time=conv.start_time.isoformat(),
                    messages=message_data,
                    token_usage=self._aggregate_token_usage(messages)
                    if messages
                    else None,
                )
                results.append((conv_data, chat_messages))

            return results

    async def get_conversation_stats(
        self,
        filters: StatsFilters,
    ) -> dict[str, dict[str, Any]]:
        """Get statistics using SQL aggregations."""
        from sqlmodel import select

        from llmling_agent.storage.models import Conversation, Message

        with Session(self.engine) as session:
            # Base query for stats
            query = (
                select(  # type: ignore[call-overload]
                    Message.model,
                    Conversation.agent_name,
                    Message.timestamp,
                    Message.total_tokens,
                    Message.prompt_tokens,
                    Message.completion_tokens,
                )
                .join(Conversation, Message.conversation_id == Conversation.id)
                .where(Message.timestamp > filters.cutoff)
            )

            if filters.agent_name:
                query = query.where(Conversation.agent_name == filters.agent_name)

            # Execute query and get raw data
            rows = [
                (
                    model,
                    agent,
                    timestamp,
                    TokenCost(
                        token_usage={
                            "total": total or 0,
                            "prompt": prompt or 0,
                            "completion": completion or 0,
                        },
                        total_cost=0.0,  # We don't store this in DB
                    )
                    if total or prompt or completion
                    else None,
                )
                for model, agent, timestamp, total, prompt, completion in session.exec(
                    query
                )
            ]

            # Use base class aggregation
            return self.aggregate_stats(rows, filters.group_by)

    def _aggregate_token_usage(self, messages: Sequence[Any]) -> TokenUsage:
        """Sum up tokens from a sequence of messages."""
        total = sum(msg.total_tokens or 0 for msg in messages)
        prompt = sum(msg.prompt_tokens or 0 for msg in messages)
        completion = sum(msg.completion_tokens or 0 for msg in messages)
        return {"total": total, "prompt": prompt, "completion": completion}
