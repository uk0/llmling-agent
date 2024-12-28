from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic_ai.models.test import TestModel
import pytest

from llmling_agent import LLMlingAgent


if TYPE_CHECKING:
    from llmling_agent.models.messages import ChatMessage


@pytest.mark.asyncio
async def test_message_chain(test_agent: LLMlingAgent[Any, str], no_tool_runtime):
    """Test that messages flow through a chain of connected agents."""
    # Create second agent
    model = TestModel(custom_result_text="Response from B")
    agent_b = LLMlingAgent[Any, str](
        runtime=no_tool_runtime,
        name="agent-b",
        model=model,
    )

    # Track all forwarded messages
    forwarded: list[tuple[LLMlingAgent[Any, Any], ChatMessage[Any]]] = []

    def collect(source: LLMlingAgent[Any, Any], msg: ChatMessage[Any]):
        forwarded.append((source, msg))

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
    # Then both agents should forward their messages
    assert len(forwarded) == 2  # noqa: PLR2004
    assert forwarded[0][0] is test_agent
    assert "I am a test response" in forwarded[0][1].content
    assert forwarded[1][0] is agent_b
    assert "Response from B" in forwarded[1][1].content
