"""Tests for the LLMling agent."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic_ai import messages


if TYPE_CHECKING:
    from llmling_agent.agent import LLMlingAgent


SIMPLE_PROMPT = "Hello, how are you?"


async def test_simple_agent_run(simple_agent: LLMlingAgent[str]) -> None:
    """Test basic agent text response."""
    result = await simple_agent.run(SIMPLE_PROMPT)
    assert isinstance(result.data, str)
    assert len(result.data) > 0


async def test_agent_message_history(simple_agent: LLMlingAgent[str]) -> None:
    """Test agent with message history."""
    history = [
        messages.UserPrompt(content="Previous message"),
        messages.ModelTextResponse(content="Previous response"),
    ]
    result = await simple_agent.run(SIMPLE_PROMPT, message_history=history)
    assert result.data
    assert simple_agent.last_run_messages
