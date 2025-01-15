"""Integration tests for chat session functionality."""

from __future__ import annotations

from pydantic_ai.models.test import TestModel
import pytest

from llmling_agent import (
    Agent,
    AgentPool,
    AgentPoolView,
    AgentsManifest,
    ChatMessage,
)


TEST_MESSAGE = "Hello, agent!"


@pytest.mark.asyncio
async def test_basic_message_response():
    """Test basic message-response flow with metadata."""
    model = TestModel(custom_result_text="Test response")
    async with Agent[None].open(model=model, name="test-agent") as agent:
        session = AgentPoolView(agent)

        response = await session.send_message(TEST_MESSAGE)

        # Verify response content and direct fields
        assert isinstance(response, ChatMessage)
        assert response.content == "Test response"
        assert response.role == "assistant"
        assert response.name == "test-agent"
        assert response.model == "test-model"
        # Verify cost info and timing
        assert response.cost_info is not None
        assert response.response_time is not None


@pytest.mark.asyncio
async def test_streaming_response():
    """Test streaming responses with chunks and metadata."""
    model = TestModel(custom_result_text="Hello world")
    async with Agent[None].open(name="test-agent", model=model) as agent:
        session = AgentPoolView(agent)

        # Get streaming response
        response_stream = await session.send_message(TEST_MESSAGE, stream=True)
        messages = [msg async for msg in response_stream]

        # Verify chunks
        assert len(messages) > 0
        assert all(isinstance(msg, ChatMessage) for msg in messages)
        # Check final message
        final_msg = messages[-1]
        assert final_msg.name == "test-agent"
        assert final_msg.model == "test-model"
        assert final_msg.cost_info is not None  # Should have cost info in final message
        # Combined content should match original
        content = "".join(msg.content for msg in messages if msg.content).strip()
        assert content == "Hello world"


@pytest.mark.asyncio
async def test_agent_forwarding():
    """Test message forwarding between agents."""
    model_1 = TestModel(custom_result_text="Main response")
    model_2 = TestModel(custom_result_text="Helper response")
    async with (
        Agent[None].open(model=model_1, name="main-agent") as main_agent,
        Agent[None].open(model=model_2, name="helper-agent") as helper_agent,
    ):
        # Set up pool and register agents
        pool = AgentPool(AgentsManifest(agents={}))
        pool.register("main", main_agent)  # Use register instead of direct assignment
        pool.register("helper", helper_agent)

        # Create session and connect agents
        session = AgentPoolView(main_agent, pool=pool)
        await session.connect_to("helper")

        # Track messages from both agents
        messages: list[ChatMessage] = []
        main_agent.message_sent.connect(messages.append)
        helper_agent.message_sent.connect(messages.append)

        # Send message and wait for forwarding
        await session.send_message(TEST_MESSAGE)
        await main_agent.complete_tasks()
        await helper_agent.complete_tasks()

        # Verify both agents responded
        assert len(messages) == 2  # noqa: PLR2004
        assert any(m.name == "main-agent" for m in messages)
        assert any(m.name == "helper-agent" for m in messages)
        assert any(m.content == "Main response" for m in messages)
        assert any(m.content == "Helper response" for m in messages)
        # Verify metrics are present
        assert all(m.cost_info is not None for m in messages)
        assert all(m.response_time is not None for m in messages)


@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling scenarios."""
    model = TestModel(custom_result_text="Test")
    async with Agent[None].open(model=model, name="test-agent") as agent:
        session = AgentPoolView(agent)

        # Test empty message
        with pytest.raises(ValueError, match="Message cannot be empty"):
            await session.send_message("")

        with pytest.raises(ValueError, match="Message cannot be empty"):
            await session.send_message("   ")
