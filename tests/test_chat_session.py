from __future__ import annotations

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock
import uuid

from pydantic_ai import messages
from pydantic_ai.result import RunResult, StreamedRunResult
import pytest

from llmling_agent.chat_session import (
    AgentChatSession,
    ChatMessage,
    ChatSessionError,
    ChatSessionManager,
)


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


# Constants for testing
DEFAULT_SESSION_ID = uuid.UUID("12345678-1234-5678-1234-567812345678")
DEFAULT_AGENT_NAME = "test-agent"
DEFAULT_MODEL = "openai:gpt-3.5-turbo"
TEST_MESSAGE = "Hello, agent!"
TEST_RESPONSE = "Hello, human!"


@pytest.fixture
def mock_agent() -> MagicMock:
    """Create a mock LLMlingAgent."""
    agent = MagicMock()
    agent.name = DEFAULT_AGENT_NAME
    # Set up default tool states
    agent.list_tools.return_value = {"tool1": True, "tool2": False}
    # Set up async methods
    agent.run = AsyncMock()
    agent.run_stream = AsyncMock()
    return agent


@pytest.fixture
def chat_session(mock_agent: MagicMock) -> AgentChatSession:
    """Create a ChatSession with a mock agent."""
    return AgentChatSession(
        agent=mock_agent,
        session_id=DEFAULT_SESSION_ID,
        model_override=DEFAULT_MODEL,
    )


@pytest.mark.asyncio
async def test_chat_session_initialization(mock_agent: MagicMock) -> None:
    """Test chat session initialization."""
    session = AgentChatSession(
        agent=mock_agent,
        session_id=DEFAULT_SESSION_ID,
        model_override=DEFAULT_MODEL,
    )

    assert session.id == DEFAULT_SESSION_ID
    assert session.metadata.agent_name == DEFAULT_AGENT_NAME
    assert session.metadata.model == DEFAULT_MODEL
    assert isinstance(session.metadata.start_time, datetime)
    assert session.metadata.tool_states == mock_agent.list_tools()


@pytest.mark.asyncio
async def test_chat_session_auto_id_generation(mock_agent: MagicMock) -> None:
    """Test that session ID is auto-generated if not provided."""
    session = AgentChatSession(agent=mock_agent)
    assert isinstance(session.id, uuid.UUID)


@pytest.mark.asyncio
async def test_send_message_normal(chat_session: AgentChatSession) -> None:
    """Test sending a normal message and getting a response."""
    # Create async mock for the run method
    mock_result = AsyncMock(spec=RunResult)
    mock_result.data = TEST_RESPONSE
    mock_result.cost.return_value = MagicMock(total_tokens=10)
    mock_result.new_messages.return_value = [
        messages.UserPrompt(content=TEST_MESSAGE),
        messages.ModelTextResponse(content=TEST_RESPONSE),
    ]
    chat_session._agent.run = AsyncMock(return_value=mock_result)  # type: ignore

    response = await chat_session.send_message(TEST_MESSAGE)

    assert isinstance(response, ChatMessage)
    assert response.content == TEST_RESPONSE
    assert response.role == "assistant"
    assert response.metadata
    assert response.metadata["tokens"] == 10  # noqa: PLR2004

    # Check call arguments but ignore timestamp
    call_args = chat_session._agent.run.await_args
    assert call_args is not None
    args, kwargs = call_args

    assert args == (TEST_MESSAGE,)
    assert kwargs["model"] == DEFAULT_MODEL
    assert len(kwargs["message_history"]) == 1
    assert kwargs["message_history"][0].content == TEST_MESSAGE
    assert kwargs["message_history"][0].role == "user"


@pytest.mark.asyncio
async def test_send_message_streaming_with_tokens(chat_session: AgentChatSession) -> None:
    """Test streaming message responses with token information."""
    from pydantic_ai import messages

    chunks = ["Hel", "lo, ", "human!"]
    stream_result = AsyncMock(spec=StreamedRunResult)
    model_responses = [messages.ModelTextResponse(content=chunk) for chunk in chunks]

    async def mock_stream() -> AsyncIterator[messages.ModelTextResponse]:
        for response in model_responses:
            yield response

    stream_result.stream = mock_stream

    # Mock cost information
    cost_mock = AsyncMock()
    cost_mock.total_tokens = 10
    cost_mock.request_tokens = 5
    cost_mock.response_tokens = 5
    stream_result.cost = AsyncMock(return_value=cost_mock)
    context_mock = AsyncMock()
    context_mock.__aenter__.return_value = stream_result
    chat_session._agent.run_stream = AsyncMock(return_value=context_mock)

    response_stream = await chat_session.send_message(TEST_MESSAGE, stream=True)

    messages = []
    async for msg in response_stream:
        messages.append(msg)

    final_msg = messages[-1]
    assert final_msg.metadata
    assert final_msg.metadata["token_usage"]["total"] == 10  # noqa: PLR2004
    assert final_msg.metadata["token_usage"]["prompt"] == 5  # noqa: PLR2004
    assert final_msg.metadata["token_usage"]["completion"] == 5  # noqa: PLR2004


@pytest.mark.asyncio
async def test_send_message_with_history(chat_session: AgentChatSession) -> None:
    """Test sending a message with existing conversation history."""
    # First message
    mock_result1 = AsyncMock(spec=RunResult)
    mock_result1.data = "First response"
    mock_result1.cost.return_value = MagicMock(total_tokens=10)

    # Set up first message history
    first_history = [
        messages.UserPrompt(content=TEST_MESSAGE),
        messages.ModelTextResponse(content="First response"),
    ]
    mock_result1.new_messages.return_value = first_history
    chat_session._agent.run = AsyncMock(return_value=mock_result1)  # type: ignore

    # Send first message
    response1 = await chat_session.send_message(TEST_MESSAGE)
    assert isinstance(response1, ChatMessage)
    assert response1.content == "First response"

    # Second message setup
    mock_result2 = AsyncMock(spec=RunResult)
    mock_result2.data = "Second response"
    mock_result2.cost.return_value = MagicMock(total_tokens=15)

    # Set up second message history
    mock_result2.new_messages.return_value = _second_history = [
        *first_history,
        messages.UserPrompt(content="Second message"),
        messages.ModelTextResponse(content="Second response"),
    ]
    chat_session._agent.run = AsyncMock(return_value=mock_result2)  # type: ignore

    # Send second message
    response2 = await chat_session.send_message("Second message")
    assert isinstance(response2, ChatMessage)
    assert response2.content == "Second response"

    # Verify history was passed correctly
    chat_session._agent.run.assert_awaited_with(
        "Second message",
        message_history=first_history,
        model=DEFAULT_MODEL,
    )


@pytest.mark.asyncio
async def test_send_message_streaming(chat_session: AgentChatSession) -> None:
    """Test streaming message responses."""
    chunks = ["Hel", "lo, ", "human!"]

    # Create async mock for the stream context
    stream_result = AsyncMock(spec=StreamedRunResult)

    # Create ModelTextResponse objects for each chunk
    model_responses = [messages.ModelTextResponse(content=chunk) for chunk in chunks]

    async def mock_stream() -> AsyncIterator[messages.ModelTextResponse]:
        for response in model_responses:
            yield response

    stream_result.stream = mock_stream

    # Explicitly set cost to None to ensure no token message
    stream_result.cost = AsyncMock(return_value=None)

    context_mock = AsyncMock()
    context_mock.__aenter__.return_value = stream_result
    chat_session._agent.run_stream = AsyncMock(return_value=context_mock)  # type: ignore

    # Get stream response
    response_stream = await chat_session.send_message(TEST_MESSAGE, stream=True)

    # Collect responses from the stream
    actual_chunks = [chunk.content async for chunk in response_stream if chunk.content]

    assert actual_chunks == chunks


@pytest.mark.asyncio
async def test_empty_message(chat_session: AgentChatSession) -> None:
    """Test handling of empty messages."""
    with pytest.raises(ValueError, match="Message cannot be empty"):
        await chat_session.send_message("")

    with pytest.raises(ValueError, match="Message cannot be empty"):
        await chat_session.send_message("   ")


@pytest.mark.asyncio
async def test_agent_error_handling(chat_session: AgentChatSession) -> None:
    """Test handling of agent errors."""
    error_msg = "Model error"
    chat_session._agent.run.side_effect = Exception(error_msg)
    chat_session._agent.run = AsyncMock(side_effect=Exception(error_msg))

    with pytest.raises(ChatSessionError, match=f"Error processing message: {error_msg}"):
        await chat_session.send_message(TEST_MESSAGE)


@pytest.mark.asyncio
async def test_configure_tools(chat_session: AgentChatSession) -> None:
    """Test tool configuration."""
    updates = {"tool1": False, "tool2": True}

    results = chat_session.configure_tools(updates)

    assert "tool1" in results
    assert "tool2" in results
    chat_session._agent.disable_tool.assert_called_once_with("tool1")
    chat_session._agent.enable_tool.assert_called_once_with("tool2")

    # Verify tool states were updated
    assert not chat_session.get_tool_states()["tool1"]
    assert chat_session.get_tool_states()["tool2"]


@pytest.mark.asyncio
async def test_configure_invalid_tool(chat_session: AgentChatSession) -> None:
    """Test configuration of non-existent tools."""
    chat_session._agent.enable_tool.side_effect = ValueError("Tool not found")

    results = chat_session.configure_tools({"invalid_tool": True})

    assert "invalid_tool" in results
    assert "error" in results["invalid_tool"]


@pytest.mark.asyncio
async def test_long_conversation(chat_session: AgentChatSession) -> None:
    """Test a longer conversation with multiple messages."""
    from unittest.mock import ANY

    messages_count = 5

    # Set up the run mock once
    chat_session._agent.run = AsyncMock()  # type: ignore

    for i in range(messages_count):
        # Create async mock result
        mock_result = AsyncMock(spec=RunResult)
        mock_result.data = f"Response {i}"
        mock_result.cost.return_value = MagicMock(total_tokens=10)

        # Create current exchange - use ANY for timestamps
        current_exchange = [
            messages.UserPrompt(content=f"Message {i}", timestamp=ANY, role="user"),
            messages.ModelTextResponse(
                content=f"Response {i}", timestamp=ANY, role="model-text-response"
            ),
        ]
        mock_result.new_messages.return_value = current_exchange

        # Set the return value for this call
        chat_session._agent.run.return_value = mock_result

        # Send message and get response
        response = await chat_session.send_message(f"Message {i}")
        assert isinstance(response, ChatMessage)
        assert response.content == f"Response {i}"

        # For the next iteration, verify history was passed correctly
        if i > 0:
            # Previous exchange should be in history
            expected_history = [
                messages.UserPrompt(
                    content=f"Message {i - 1}", timestamp=ANY, role="user"
                ),
                messages.ModelTextResponse(
                    content=f"Response {i - 1}", timestamp=ANY, role="model-text-response"
                ),
                messages.UserPrompt(
                    content=f"Message {i}", timestamp=ANY, role="user"
                ),  # Current message
            ]

            chat_session._agent.run.assert_awaited_with(
                f"Message {i}",
                message_history=expected_history,
                model=DEFAULT_MODEL,
            )

    # Verify total number of interactions
    assert chat_session._agent.run.await_count == messages_count


@pytest.mark.asyncio
async def test_concurrent_messages(chat_session: AgentChatSession) -> None:
    """Test handling of concurrent message sending."""

    async def slow_response(content: str, **kwargs: Any) -> RunResult:
        await asyncio.sleep(0.1)
        mock_result = AsyncMock(spec=RunResult)
        mock_result.data = f"Response to: {content}"
        mock_result.cost.return_value = MagicMock(total_tokens=10)
        mock_result.new_messages.return_value = [
            messages.UserPrompt(content=content),
            messages.ModelTextResponse(content=f"Response to: {content}"),
        ]
        return mock_result

    chat_session._agent.run = AsyncMock(side_effect=slow_response)  # type: ignore

    # Send multiple messages concurrently
    message_texts = ["First", "Second", "Third"]
    tasks = [chat_session.send_message(msg, stream=False) for msg in message_texts]

    # Gather responses
    responses = await asyncio.gather(*tasks)

    # Verify all messages were processed
    assert len(responses) == len(message_texts)
    for response, original_msg in zip(responses, message_texts):
        assert isinstance(response, ChatMessage)
        assert response.role == "assistant"
        assert response.content == f"Response to: {original_msg}"

    # Verify all calls were made
    assert chat_session._agent.run.await_count == len(message_texts)


@pytest.mark.asyncio
async def test_message_after_tool_update(chat_session: AgentChatSession) -> None:
    """Test sending messages after tool configuration changes."""
    # First configure tools
    chat_session.configure_tools({"tool1": False})

    # Then send message
    mock_result = AsyncMock(spec=RunResult)
    mock_result.data = TEST_RESPONSE
    mock_result.cost.return_value = MagicMock(total_tokens=10)
    mock_result.new_messages.return_value = [
        messages.UserPrompt(content=TEST_MESSAGE),
        messages.ModelTextResponse(content=TEST_RESPONSE),
    ]
    chat_session._agent.run = AsyncMock(return_value=mock_result)  # type: ignore

    response = await chat_session.send_message(TEST_MESSAGE)

    assert isinstance(response, ChatMessage)
    assert response.content == TEST_RESPONSE
    assert response.role == "assistant"

    # Verify tool state persisted
    assert not chat_session.get_tool_states()["tool1"]


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
    mock_agent.enable_tool.assert_called_with("tool1")
    mock_agent.disable_tool.assert_called_with("tool1")
