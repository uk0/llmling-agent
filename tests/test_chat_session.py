from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

from llmling import Config, RuntimeConfig
from pydantic_ai.models.test import TestModel
import pytest

from llmling_agent.agent.agent import LLMlingAgent
from llmling_agent.chat_session import (
    AgentChatSession,
    ChatSessionError,
    ChatSessionManager,
)
from llmling_agent.delegation.pool import AgentPool
from llmling_agent.models.agents import AgentsManifest
from llmling_agent.models.messages import ChatMessage
from llmling_agent_cli.chat_session.session import InteractiveSession


# Constants for testing
DEFAULT_MODEL = "openai:gpt-3.5-turbo"
TEST_MESSAGE = "Hello, agent!"
TEST_RESPONSE = "Hello, human!"


@pytest.fixture
def mock_agent() -> MagicMock:
    """Create a mock LLMlingAgent."""
    agent = MagicMock()
    agent.name = "test-agent"
    # Set up default tool states
    agent.tools.list_tools.return_value = {"tool1": True, "tool2": False}
    # Set up async methods
    agent.run = AsyncMock()
    agent.run_stream = AsyncMock()
    return agent


@pytest.fixture
async def chat_session(mock_agent) -> AgentChatSession:
    """Provide a test chat session."""
    session = AgentChatSession(agent=mock_agent)
    await session.initialize()
    return session


@pytest.mark.asyncio
async def test_send_message_normal():
    """Test normal message sending."""
    async with RuntimeConfig.open(Config()) as runtime:
        agent: LLMlingAgent[Any, Any] = LLMlingAgent(
            runtime, model=TestModel(custom_result_text=TEST_RESPONSE), name="test-agent"
        )
        session = AgentChatSession(agent)
        await session.initialize()

        response = await session.send_message(TEST_MESSAGE)
        assert isinstance(response, ChatMessage)
        assert response.content == TEST_RESPONSE
        assert response.role == "assistant"
        assert response.metadata
        assert response.metadata.name == agent.name


@pytest.mark.asyncio
async def test_send_message_streaming_with_tokens(chat_session: AgentChatSession):
    """Test streaming message responses with token information."""
    async with RuntimeConfig.open(Config()) as runtime:
        # Create agent with TestModel that returns chunks
        agent: LLMlingAgent[Any, Any] = LLMlingAgent(
            runtime,
            model=TestModel(custom_result_text="Hello, human!"),
            name="test-agent",
        )
        session = AgentChatSession(agent)
        await session.initialize()

        # Get streaming response
        response_stream = await session.send_message(TEST_MESSAGE, stream=True)
        messages = [msg async for msg in response_stream]

        # Verify streaming messages
        assert len(messages) > 1  # Should get multiple chunks
        # Last message should have metadata
        final_msg = messages[-1]
        assert final_msg.metadata
        assert final_msg.metadata.model == "test-model"


@pytest.mark.asyncio
async def test_send_message_streaming(chat_session: AgentChatSession):
    """Test streaming message responses."""
    async with RuntimeConfig.open(Config()) as runtime:
        agent: LLMlingAgent[Any, Any] = LLMlingAgent(
            runtime, model=TestModel(custom_result_text="Hello world"), name="test-agent"
        )
        session = AgentChatSession(agent)
        await session.initialize()

        response_stream = await session.send_message(TEST_MESSAGE, stream=True)
        responses = [msg async for msg in response_stream]

        # TestModel returns response in chunks
        assert len(responses) > 0
        assert all(isinstance(r, ChatMessage) for r in responses)
        assert "".join(r.content for r in responses if r.content).strip() == "Hello world"


@pytest.mark.asyncio
async def test_empty_message(chat_session: AgentChatSession):
    """Test handling of empty messages."""
    with pytest.raises(ValueError, match="Message cannot be empty"):
        await chat_session.send_message("")

    with pytest.raises(ValueError, match="Message cannot be empty"):
        await chat_session.send_message("   ")


@pytest.mark.asyncio
async def test_agent_error_handling(chat_session: AgentChatSession):
    """Test handling of agent errors."""
    error_msg = "Model error"
    chat_session._agent.run.side_effect = Exception(error_msg)  # type: ignore
    chat_session._agent.run = AsyncMock(side_effect=Exception(error_msg))  # type: ignore

    with pytest.raises(ChatSessionError, match=f"Error processing message: {error_msg}"):
        await chat_session.send_message(TEST_MESSAGE)


@pytest.mark.asyncio
async def test_configure_tools(chat_session: AgentChatSession):
    """Test tool configuration."""
    updates = {"tool1": False, "tool2": True}

    results = chat_session.configure_tools(updates)

    assert "tool1" in results
    assert "tool2" in results
    chat_session._agent.tools.disable_tool.assert_called_once_with("tool1")  # type: ignore
    chat_session._agent.tools.enable_tool.assert_called_once_with("tool2")  # type: ignore

    # Verify tool states were updated
    assert not chat_session.get_tool_states()["tool1"]
    assert chat_session.get_tool_states()["tool2"]


@pytest.mark.asyncio
async def test_configure_invalid_tool(chat_session: AgentChatSession):
    """Test configuration of non-existent tools."""
    chat_session._agent.tools.enable_tool.side_effect = ValueError("Tool not found")  # type: ignore

    results = chat_session.configure_tools({"invalid_tool": True})

    assert "invalid_tool" in results
    assert "error" in results["invalid_tool"]


@pytest.mark.asyncio
async def test_long_conversation(chat_session: AgentChatSession):
    """Test a longer conversation with multiple messages."""
    async with RuntimeConfig.open(Config()) as runtime:
        responses = []
        for i in range(5):
            # Create new TestModel for each iteration with specific response
            agent: LLMlingAgent[Any, Any] = LLMlingAgent(
                runtime,
                model=TestModel(custom_result_text=f"Response {i}"),
                name="test-agent",
            )
            session = AgentChatSession(agent)
            await session.initialize()

            response = await session.send_message(f"Message {i}")
            responses.append(response)
            assert response.content == f"Response {i}"

        assert len(responses) == 5  # noqa: PLR2004


@pytest.mark.asyncio
async def test_concurrent_messages(chat_session: AgentChatSession):
    """Test handling of concurrent message sending."""
    async with RuntimeConfig.open(Config()) as runtime:
        # Create separate agents with different responses
        agents = []
        message_texts = ["First", "Second", "Third"]

        for msg in message_texts:
            agent: LLMlingAgent[Any, Any] = LLMlingAgent(
                runtime,
                model=TestModel(custom_result_text=f"Response to: {msg}"),
                name=f"test-agent-{msg}",
            )
            agents.append(agent)

        # Run messages through respective agents
        responses = []
        for agent, msg in zip(agents, message_texts):
            session = AgentChatSession(agent)
            await session.initialize()
            response = await session.send_message(msg)
            responses.append(response)

        # Verify responses
        assert len(responses) == len(message_texts)
        for response, original_msg in zip(responses, message_texts):
            assert response.content == f"Response to: {original_msg}"


@pytest.mark.asyncio
async def test_message_after_tool_update():
    """Test sending messages after tool configuration changes."""
    async with RuntimeConfig.open(Config()) as runtime:
        # Create agent with TestModel
        agent: LLMlingAgent[Any, Any] = LLMlingAgent(
            runtime, model=TestModel(custom_result_text=TEST_RESPONSE), name="test-agent"
        )

        # Register a test tool
        def test_tool():
            """Test tool."""
            return "test result"

        agent.tools.register_tool(test_tool, enabled=True, source="dynamic")

        session = AgentChatSession(agent)
        await session.initialize()

        # Configure tools
        session.configure_tools({"test_tool": False})  # Use actual registered name

        # Send message
        response = await session.send_message(TEST_MESSAGE)
        assert response.content == TEST_RESPONSE

        # Verify tool state persisted
        assert not session.get_tool_states()["test_tool"]


@pytest.mark.asyncio
async def test_chat_session_with_tools(mock_agent):
    """Test chat session managing tool states and history."""
    # Use the mock_agent fixture which already has tools set up
    manager = ChatSessionManager()
    session = await manager.create_session(mock_agent)

    # Test initial tool states
    tool_states = session.get_tool_states()
    assert "tool1" in tool_states
    assert "tool2" in tool_states
    assert tool_states["tool1"] is True
    assert tool_states["tool2"] is False

    # Disable a tool and verify
    session.configure_tools({"tool1": False})
    assert not session.get_tool_states()["tool1"]

    # Re-enable and verify
    session.configure_tools({"tool1": True})
    assert session.get_tool_states()["tool1"]

    # Verify that mock_agent's enable/disable methods were called
    mock_agent.tools.enable_tool.assert_called_with("tool1")
    mock_agent.tools.disable_tool.assert_called_with("tool1")


@pytest.mark.asyncio
async def test_message_forwarding_in_cli():
    """Test that forwarded messages are displayed properly."""
    async with RuntimeConfig.open(Config()) as runtime:
        main_agent: LLMlingAgent[Any, Any] = LLMlingAgent(
            runtime,
            model=TestModel(custom_result_text="Main response"),
            name="main-agent",
        )
        poet_agent: LLMlingAgent[Any, Any] = LLMlingAgent(
            runtime,
            model=TestModel(custom_result_text="Poem response"),
            name="poet-agent",
        )

        pool = AgentPool(AgentsManifest(agents={}))
        pool.agents["main"] = main_agent
        pool.agents["poet"] = poet_agent

        session = InteractiveSession(main_agent, pool=pool)
        session._chat_session = await session._session_manager.create_session(
            main_agent, pool=pool, wait_chain=True
        )

        # Connect to messages directly
        messages_received = []
        main_agent.message_sent.connect(lambda msg: messages_received.append(msg))
        poet_agent.message_sent.connect(lambda msg: messages_received.append(msg))

        await session._chat_session.connect_to("poet")
        await session._chat_session.send_message("Write a poem")

        # Wait for forwarding
        await main_agent.complete_tasks()
        await poet_agent.complete_tasks()

        assert len(messages_received) == 2  # noqa: PLR2004
        print(messages_received)
        assert any(m.metadata.name == "main-agent" for m in messages_received)
        assert any(m.metadata.name == "poet-agent" for m in messages_received)
