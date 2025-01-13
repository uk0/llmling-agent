from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from sqlmodel import Session, SQLModel, create_engine, delete

from llmling_agent.models.messages import TokenCost
from llmling_agent.utils.parse_time import parse_time_period
from llmling_agent_storage.models import QueryFilters, StatsFilters
from llmling_agent_storage.sql_provider import Conversation, Message, SQLModelProvider


# Reference time for all tests
BASE_TIME = datetime(2024, 1, 1, 12, 0)  # noon on Jan 1, 2024

# Create in-memory database for testing
engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
)

# Create all tables
SQLModel.metadata.create_all(engine)


@pytest.fixture(autouse=True)
def cleanup_database():
    """Clean up the database before each test."""
    SQLModel.metadata.create_all(engine)  # Ensure tables exist
    with Session(engine) as session:
        # Delete all messages first (due to foreign key)
        session.execute(delete(Message))
        session.execute(delete(Conversation))
        session.commit()


@pytest.fixture
async def sample_data(cleanup_database: None):
    """Create sample conversation data."""
    with Session(engine) as session:
        # Create two conversations
        start = BASE_TIME - timedelta(hours=1)  # 11:00
        conv1 = Conversation(id="conv1", agent_name="test_agent", start_time=start)
        start = BASE_TIME - timedelta(hours=2)  # 10:00
        conv2 = Conversation(id="conv2", agent_name="other_agent", start_time=start)
        session.add(conv1)
        session.add(conv2)
        session.commit()

        # Create provider for async operations
        provider = SQLModelProvider(engine)

        # Messages to add
        test_data = [
            (
                "conv1",  # conversation_id
                "Hello",  # content
                "user",  # role
                "user",  # name
                "gpt-4",  # model
                TokenCost(
                    token_usage={"total": 10, "prompt": 5, "completion": 5},
                    total_cost=0.001,
                ),  # cost_info
                BASE_TIME - timedelta(hours=1),  # message time
            ),
            (
                "conv1",
                "Hi there!",
                "assistant",
                "test_agent",
                "gpt-4",
                TokenCost(
                    token_usage={"total": 20, "prompt": 10, "completion": 10},
                    total_cost=0.002,
                ),
                BASE_TIME - timedelta(hours=1),
            ),
            (
                "conv2",
                "Testing",
                "user",
                "user",
                "gpt-3.5-turbo",
                TokenCost(
                    token_usage={"total": 15, "prompt": 7, "completion": 8},
                    total_cost=0.0015,
                ),
                BASE_TIME - timedelta(hours=2),
            ),
        ]

        # Add messages using the provider's method signature
        for conv_id, content, role, name, model, cost_info, msg_time in test_data:
            await provider.log_message(
                conversation_id=conv_id,
                content=content,
                role=role,
                name=name,
                model=model,
                cost_info=cost_info,
                response_time=None,
                forwarded_from=None,
            )


@pytest.fixture
async def provider():
    """Create SQLModelProvider instance."""
    return SQLModelProvider(engine)


def test_parse_time_period():
    """Test time period parsing."""
    assert parse_time_period("1h") == timedelta(hours=1)
    assert parse_time_period("2d") == timedelta(days=2)
    assert parse_time_period("1w") == timedelta(weeks=1)


@pytest.mark.asyncio
async def test_get_conversations(provider: SQLModelProvider, sample_data: None):
    """Test conversation retrieval with filters."""
    # Get all conversations
    filters = QueryFilters()
    results = await provider.get_conversations(filters)
    assert len(results) == 2  # noqa: PLR2004

    # Filter by agent
    filters = QueryFilters(agent_name="test_agent")
    results = await provider.get_conversations(filters)
    assert len(results) == 1
    conv, msgs = results[0]
    assert conv["agent"] == "test_agent"  # Note: Changed from agent_name to agent
    assert len(msgs) == 2  # noqa: PLR2004

    # Filter by time
    filters = QueryFilters(since=BASE_TIME - timedelta(hours=1.5))
    results = await provider.get_conversations(filters)
    assert len(results) == 1


@pytest.mark.asyncio
async def test_get_conversation_stats(provider: SQLModelProvider, sample_data: None):
    """Test statistics retrieval and aggregation."""
    cutoff = BASE_TIME - timedelta(hours=3)
    filters = StatsFilters(cutoff=cutoff, group_by="model")
    stats = await provider.get_conversation_stats(filters)

    # Check model grouping
    assert "gpt-4" in stats
    assert stats["gpt-4"]["messages"] == 2  # noqa: PLR2004
    assert stats["gpt-4"]["total_tokens"] == 30  # noqa: PLR2004
    assert "gpt-3.5-turbo" in stats
    assert stats["gpt-3.5-turbo"]["messages"] == 1


@pytest.mark.asyncio
async def test_complex_filtering(provider: SQLModelProvider, sample_data: None):
    """Test combined filtering capabilities."""
    filters = QueryFilters(
        agent_name="test_agent",
        model="gpt-4",
        since=BASE_TIME - timedelta(hours=1.5),
        query="Hello",
    )
    results = await provider.get_conversations(filters)
    assert len(results) == 1
    conv, msgs = results[0]
    assert conv["agent"] == "test_agent"
    assert any(msg.content == "Hello" for msg in msgs)
    assert all(msg.model == "gpt-4" for msg in msgs)


@pytest.mark.asyncio
async def test_filtered_conversations(provider: SQLModelProvider, sample_data: None):
    """Test high-level filtered conversation helper."""
    results = await provider.get_filtered_conversations(
        agent_name="test_agent", period="2h", include_tokens=True
    )
    assert len(results) == 1
    conv = results[0]
    assert conv["agent"] == "test_agent"
    assert len(conv["messages"]) == 2  # noqa: PLR2004
    assert conv["token_usage"] is not None
    assert conv["token_usage"]["total"] == 30  # 10 + 20 tokens  # noqa: PLR2004
