"""Integration tests for chat session functionality."""

from __future__ import annotations

from typing import Any

from llmling import Config, RuntimeConfig
from llmling.tools import ToolError
from pydantic_ai.models.test import TestModel
import pytest

from llmling_agent import LLMlingAgent
from llmling_agent.chat_session import AgentChatSession
from llmling_agent.delegation.pool import AgentPool
from llmling_agent.models.agents import AgentsManifest
from llmling_agent.models.messages import ChatMessage


TEST_MESSAGE = "Hello, agent!"


@pytest.mark.asyncio
async def test_basic_message_response():
    """Test basic message-response flow with metadata."""
    async with RuntimeConfig.open(Config()) as runtime:
        agent = LLMlingAgent[Any, Any](
            runtime,
            model=TestModel(custom_result_text="Test response"),
            name="test-agent",
        )
        session = AgentChatSession(agent)
        await session.initialize()

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
    async with RuntimeConfig.open(Config()) as runtime:
        agent = LLMlingAgent[Any, Any](
            runtime, model=TestModel(custom_result_text="Hello world"), name="test-agent"
        )
        session = AgentChatSession(agent)
        await session.initialize()

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
async def test_tool_management():
    """Test tool registration, enabling/disabling, and usage."""
    async with RuntimeConfig.open(Config()) as runtime:
        agent = LLMlingAgent[Any, Any](
            runtime, model=TestModel(custom_result_text="Tool test"), name="test-agent"
        )

        # Register test tools
        def tool1():
            return "tool1 result"

        def tool2():
            return "tool2 result"

        agent.tools.register_tool(tool1, enabled=True, source="dynamic")
        agent.tools.register_tool(tool2, enabled=True, source="dynamic")

        session = AgentChatSession(agent)
        await session.initialize()

        # Test tool configuration
        initial_states = session.get_tool_states()
        assert initial_states["tool1"] is True
        assert initial_states["tool2"] is True

        # Disable tool and verify
        session.configure_tools({"tool1": False})
        assert not session.get_tool_states()["tool1"]
        assert session.get_tool_states()["tool2"]


@pytest.mark.asyncio
async def test_agent_forwarding():
    """Test message forwarding between agents."""
    async with RuntimeConfig.open(Config()) as runtime:
        # Create agents with different responses
        main_agent = LLMlingAgent[Any, Any](
            runtime,
            model=TestModel(custom_result_text="Main response"),
            name="main-agent",
        )
        helper_agent = LLMlingAgent[Any, Any](
            runtime,
            model=TestModel(custom_result_text="Helper response"),
            name="helper-agent",
        )

        # Set up pool and forwarding
        pool = AgentPool(AgentsManifest(agents={}))
        pool.agents["main"] = main_agent
        pool.agents["helper"] = helper_agent

        # Create session and connect agents
        session = AgentChatSession(main_agent, pool=pool)
        await session.initialize()
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
    async with RuntimeConfig.open(Config()) as runtime:
        agent = LLMlingAgent[Any, Any](
            runtime, model=TestModel(custom_result_text="Test"), name="test-agent"
        )
        session = AgentChatSession(agent)
        await session.initialize()

        # Test empty message
        with pytest.raises(ValueError, match="Message cannot be empty"):
            await session.send_message("")

        with pytest.raises(ValueError, match="Message cannot be empty"):
            await session.send_message("   ")

        # Test handling of invalid tool operations
        with pytest.raises(ToolError):
            session.configure_tools({"nonexistent": True})
