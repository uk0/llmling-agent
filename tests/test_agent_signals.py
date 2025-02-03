from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic_ai.models.test import TestModel
import pytest

from llmling_agent import Agent


if TYPE_CHECKING:
    from llmling_agent.messaging.messages import ChatMessage


@pytest.mark.asyncio
async def test_message_chain():
    """Test that message chain tracks transformations correctly."""
    async with Agent[None](name="agent-a", model="test") as agent_a:  # noqa: SIM117
        async with Agent[None](name="agent-b", model="test") as agent_b:
            async with Agent[None](name="agent-c", model="test") as agent_c:
                # Connect chain
                agent_a.connect_to(agent_b)
                agent_b.connect_to(agent_c)

                # When A processes a new message
                result_a = await agent_a.run("Start")
                assert result_a.forwarded_from == []  # New message, empty chain

                # When B processes A's message
                result_b = await agent_b.run(result_a)
                assert result_b.forwarded_from == [agent_a.name]  # Chain includes A

                # When C processes B's message
                result_c = await agent_c.run(result_b)
                assert result_c.forwarded_from == [agent_a.name, agent_b.name]


@pytest.mark.asyncio
async def test_run_result_not_modified_by_connections():
    """Test that the message returned by run() isn't modified by connections."""
    # Create two agents
    model = TestModel(custom_result_text="Response from B")
    async with Agent[None](name="agent-a", model=model) as agent_a:  # noqa: SIM117
        async with Agent[None](name="agent-b", model=model) as agent_b:
            # Connect A to B
            agent_a.connect_to(agent_b)

            # When A runs
            result = await agent_a.run("Test message")

            # Then the returned message should only contain A as source
            assert result.forwarded_from == [], (
                "run() result should have empty forwarded_from"
            )
            # or possibly just [agent_a.name] if we decide that's the expected behavior

            # While messages received by B should have the full chain
            def collect_b(msg: ChatMessage[Any], _prompt: str | None = None):
                assert msg.forwarded_from == ["agent-a"], (
                    "Forwarded message should contain chain"
                )

            agent_b.outbox.connect(collect_b)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
