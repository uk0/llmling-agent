from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from unittest.mock import ANY, AsyncMock, MagicMock, patch

from pydantic_ai.messages import (
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.result import RunResult, StreamedRunResult
import pytest

from llmling_agent.chat_session import (
    AgentChatSession,
    ChatSessionError,
    ChatSessionManager,
)
from llmling_agent.models.messages import ChatMessage, TokenAndCostResult


if TYPE_CHECKING:
    from collections.abc import AsyncIterator


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
    session = AgentChatSession(agent=mock_agent, model_override=DEFAULT_MODEL)
    await session.initialize()  # Add this line
    return session


@pytest.mark.asyncio
async def test_send_message_normal(chat_session: AgentChatSession) -> None:
    """Test normal message sending."""
    mock_result = AsyncMock()
    mock_usage = MagicMock()
    mock_usage.total_tokens = 10
    mock_usage.request_tokens = 5
    mock_usage.response_tokens = 5
    mock_result.usage.return_value = mock_usage
    mock_result.data = TEST_RESPONSE

    # Mock the token cost calculation
    mock_token_result = TokenAndCostResult(
        token_usage={"total": 10, "prompt": 5, "completion": 5}, cost_usd=0.0001
    )
    with patch(
        "llmling_agent.chat_session.base.extract_token_usage_and_cost",
        AsyncMock(return_value=mock_token_result),
    ):
        user_part: ModelRequestPart = UserPromptPart(content=TEST_MESSAGE)
        response_part: ModelResponsePart = TextPart(content=TEST_RESPONSE)
        mock_result.new_messages.return_value = [
            ModelRequest(parts=[user_part]),
            ModelResponse(parts=[response_part]),
        ]
        chat_session._agent.run = AsyncMock(return_value=mock_result)  # type: ignore

        response = await chat_session.send_message(TEST_MESSAGE)
        assert isinstance(response, ChatMessage)
        assert response.content == TEST_RESPONSE
        assert response.role == "assistant"
        assert response.metadata
        assert response.token_usage
        assert response.token_usage["total"] == 10  # noqa: PLR2004


@pytest.mark.asyncio
async def test_send_message_streaming_with_tokens(chat_session: AgentChatSession) -> None:
    """Test streaming message responses with token information."""
    chunks = ["Hel", "lo, ", "human!"]
    stream_result = AsyncMock(spec=StreamedRunResult)

    async def mock_stream() -> AsyncIterator[str]:
        for chunk in chunks:
            yield chunk

    stream_result.stream = mock_stream
    mock_usage = MagicMock()
    mock_usage.total_tokens = 10
    mock_usage.request_tokens = 5
    mock_usage.response_tokens = 5
    stream_result.usage = MagicMock(return_value=mock_usage)
    # Mock the token cost calculation
    mock_token_result = TokenAndCostResult(
        token_usage={"total": 10, "prompt": 5, "completion": 5}, cost_usd=0.0001
    )
    with patch(
        "llmling_agent.chat_session.base.extract_token_usage_and_cost",
        AsyncMock(return_value=mock_token_result),
    ):
        context_mock = AsyncMock()
        context_mock.__aenter__.return_value = stream_result
        chat_session._agent.run_stream = AsyncMock(return_value=context_mock)  # type: ignore

        response_stream = await chat_session.send_message(TEST_MESSAGE, stream=True)
        messages = [msg async for msg in response_stream]
        final_msg = messages[-1]
        assert final_msg.metadata
        assert final_msg.metadata.token_usage
        assert final_msg.metadata.token_usage["total"] == 10  # noqa: PLR2004


@pytest.mark.asyncio
async def test_send_message_with_history(chat_session: AgentChatSession) -> None:
    """Test sending a message with existing conversation history."""
    # First message
    mock_result1 = AsyncMock(spec=RunResult)
    mock_result1.data = "First response"
    mock_result1.usage.return_value = MagicMock(total_tokens=10)

    # Set up first message history
    user = UserPromptPart(content=TEST_MESSAGE)
    text = TextPart(content="First response")
    first_history = [ModelRequest(parts=[user]), ModelResponse(parts=[text])]
    mock_result1.new_messages.return_value = first_history
    chat_session._agent.run = AsyncMock(return_value=mock_result1)  # type: ignore

    # Send first message
    response1 = await chat_session.send_message(TEST_MESSAGE)
    assert isinstance(response1, ChatMessage)
    assert response1.content == "First response"

    # Second message setup
    mock_result2 = AsyncMock(spec=RunResult)
    mock_result2.data = "Second response"
    mock_result2.usage.return_value = MagicMock(total_tokens=15)

    # Set up second message history
    second_history = [
        *first_history,
        ModelRequest(parts=[UserPromptPart(content="Second message")]),
        ModelResponse(parts=[TextPart(content="Second response")]),
    ]
    mock_result2.new_messages.return_value = second_history
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
    stream_result = AsyncMock(spec=StreamedRunResult)

    async def mock_stream() -> AsyncIterator[str]:
        for chunk in chunks:
            yield chunk

    stream_result.stream = mock_stream

    # Create a MagicMock for cost instead of AsyncMock
    mock_usage = MagicMock()
    mock_usage.total_tokens = None
    mock_usage.request_tokens = None
    mock_usage.response_tokens = None
    stream_result.usage = MagicMock(return_value=mock_usage)

    context_mock = AsyncMock()
    context_mock.__aenter__.return_value = stream_result
    chat_session._agent.run_stream = AsyncMock(return_value=context_mock)  # type: ignore

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
    chat_session._agent.run.side_effect = Exception(error_msg)  # type: ignore
    chat_session._agent.run = AsyncMock(side_effect=Exception(error_msg))  # type: ignore

    with pytest.raises(ChatSessionError, match=f"Error processing message: {error_msg}"):
        await chat_session.send_message(TEST_MESSAGE)


@pytest.mark.asyncio
async def test_configure_tools(chat_session: AgentChatSession) -> None:
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
async def test_configure_invalid_tool(chat_session: AgentChatSession) -> None:
    """Test configuration of non-existent tools."""
    chat_session._agent.tools.enable_tool.side_effect = ValueError("Tool not found")  # type: ignore

    results = chat_session.configure_tools({"invalid_tool": True})

    assert "invalid_tool" in results
    assert "error" in results["invalid_tool"]


@pytest.mark.asyncio
async def test_long_conversation(chat_session: AgentChatSession) -> None:
    """Test a longer conversation with multiple messages."""
    messages_count = 5

    # Set up the run mock once
    chat_session._agent.run = AsyncMock()  # type: ignore

    for i in range(messages_count):
        # Create async mock result
        mock_result = AsyncMock(spec=RunResult)
        mock_result.data = f"Response {i}"
        mock_result.usage.return_value = MagicMock(total_tokens=10)

        # Create current exchange
        user_message = ModelRequest(
            parts=[UserPromptPart(content=f"Message {i}", timestamp=ANY)]
        )
        assistant_message = ModelResponse(parts=[TextPart(content=f"Response {i}")])

        # Update history for next iteration
        current_exchange = [user_message, assistant_message]
        mock_result.new_messages.return_value = current_exchange
        _history = current_exchange  # Store for next iteration

        # Set the return value for this call
        chat_session._agent.run.return_value = mock_result

        # Send message and get response
        response = await chat_session.send_message(f"Message {i}")
        assert isinstance(response, ChatMessage)
        assert response.content == f"Response {i}"

    # Verify total number of interactions
    assert chat_session._agent.run.await_count == messages_count


@pytest.mark.asyncio
async def test_concurrent_messages(chat_session: AgentChatSession) -> None:
    """Test handling of concurrent message sending."""

    async def slow_response(content: str, **kwargs: Any) -> RunResult:
        await asyncio.sleep(0.1)
        mock_result = AsyncMock(spec=RunResult)
        mock_result.data = f"Response to: {content}"
        mock_result.usage.return_value = MagicMock(total_tokens=10)

        # Create properly typed parts
        user_part: ModelRequestPart = UserPromptPart(content=content)
        response_part: ModelResponsePart = TextPart(content=f"Response to: {content}")

        mock_result.new_messages.return_value = [
            ModelRequest(parts=[user_part]),
            ModelResponse(parts=[response_part]),
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
    mock_result.usage.return_value = MagicMock(total_tokens=10)

    # Create properly typed parts
    user_part: ModelRequestPart = UserPromptPart(content=TEST_MESSAGE)
    response_part: ModelResponsePart = TextPart(content=TEST_RESPONSE)

    mock_result.new_messages.return_value = [
        ModelRequest(parts=[user_part]),
        ModelResponse(parts=[response_part]),
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
    mock_agent.tools.enable_tool.assert_called_with("tool1")
    mock_agent.tools.disable_tool.assert_called_with("tool1")
