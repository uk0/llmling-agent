from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic_ai.models.test import TestModel
import pytest

from llmling_agent import Agent


if TYPE_CHECKING:
    from llmling_agent.models.messages import ChatMessage


@pytest.mark.asyncio
async def test_message_chain(test_agent: Agent[Any, str], no_tool_runtime):
    """Test that messages flow through a chain of connected agents."""
    # Create second agent
    model = TestModel(custom_result_text="Response from B")
    agent_b = Agent[Any, str](runtime=no_tool_runtime, name="agent-b", model=model)

    # Track all forwarded messages
    forwarded: list[tuple[str, ChatMessage[Any], str | None]] = []

    def collect(msg: ChatMessage[Any], prompt: str | None = None):
        sender = msg.forwarded_from[-1] if msg.forwarded_from else None
        if sender is None:
            error = "Message without sender information"
            raise RuntimeError(error)
        forwarded.append((sender, msg, prompt))

    # Connect both agents' forwards to our collector
    test_agent.outbox.connect(collect)
    agent_b.outbox.connect(collect)

    # Connect the chain
    test_agent.pass_results_to(agent_b)

    # When test_agent sends a message
    await test_agent.run("Start message")
    await agent_b.complete_tasks()

    # Then both messages should be forwarded
    assert len(forwarded) == 2  # noqa: PLR2004

    # Check first message (from test_agent)
    assert forwarded[0][0] == test_agent.name
    assert "I am a test response" in forwarded[0][1].content
    assert forwarded[0][2] is None  # no prompt

    # Check second message (from agent_b)
    assert forwarded[1][0] == agent_b.name
    assert "Response from B" in forwarded[1][1].content
    assert forwarded[1][2] is None  # no prompt
