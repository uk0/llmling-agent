from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.result import RunResult
import pytest

from llmling_agent.agent.agent import LLMlingAgent


@pytest.mark.asyncio
async def test_conversation_history_management():
    """Test ConversationManager maintains proper message history."""
    async with LLMlingAgent[Any, str].open() as agent:
        # First message setup
        mock_result1 = AsyncMock(spec=RunResult)
        mock_result1.data = "First response"
        mock_result1.usage.return_value = MagicMock(total_tokens=10)

        # Set up first message history
        first_history = [
            ModelRequest(parts=[UserPromptPart(content="First message")]),
            ModelResponse(parts=[TextPart(content="First response")]),
        ]
        mock_result1.new_messages.return_value = first_history
        mock_result1.all_messages.return_value = first_history
        agent._pydantic_agent.run = AsyncMock(return_value=mock_result1)  # type: ignore

        # Run first message
        result1 = await agent.run("First message")
        assert str(result1.data) == "First response"

        # Verify history was stored
        history = agent.conversation.get_history()
        assert len(history) == 2  # Request and Response  # noqa: PLR2004

        # Second message setup
        mock_result2 = AsyncMock(spec=RunResult)
        mock_result2.data = "Second response"
        mock_result2.usage.return_value = MagicMock(total_tokens=15)

        second_history = [
            *first_history,
            ModelRequest(parts=[UserPromptPart(content="Second message")]),
            ModelResponse(parts=[TextPart(content="Second response")]),
        ]
        mock_result2.new_messages.return_value = second_history
        mock_result2.all_messages.return_value = second_history
        agent._pydantic_agent.run = AsyncMock(return_value=mock_result2)  # type: ignore

        # Run second message
        result2 = await agent.run("Second message")
        assert str(result2.data) == "Second response"

        # Verify complete history
        final_history = agent.conversation.get_history()
        assert len(final_history) == 4  # Both requests and responses  # noqa: PLR2004
