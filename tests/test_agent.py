"""Tests for the LLMling agent."""

from __future__ import annotations

import asyncio

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models.test import TestModel
import pytest

from llmling_agent import Agent, AgentPool, ChatMessage


SIMPLE_PROMPT = "Hello, how are you?"
TEST_RESPONSE = "I am a test response"


@pytest.mark.asyncio
async def test_simple_agent_run(test_agent: Agent[None]):
    """Test basic agent text response."""
    result = await test_agent.run(SIMPLE_PROMPT)
    assert isinstance(result.data, str)
    assert result.data == TEST_RESPONSE
    assert result.cost_info is not None


@pytest.mark.asyncio
async def test_agent_message_history(test_agent: Agent[None]):
    """Test agent with message history."""
    history = [
        ChatMessage(content="Previous message", role="user"),
        ChatMessage(content="Previous response", role="assistant"),
    ]
    test_agent.conversation.set_history(history)
    result = await test_agent.run(SIMPLE_PROMPT)
    assert result.data == TEST_RESPONSE
    assert test_agent.conversation.last_run_messages
    assert len(test_agent.conversation.last_run_messages) == 2  # noqa: PLR2004


@pytest.mark.asyncio
async def test_agent_streaming(test_agent: Agent[None]):
    """Test agent streaming response."""
    stream_ctx = test_agent.run_stream(SIMPLE_PROMPT)
    async with stream_ctx as stream:
        collected = [str(message) async for message in stream.stream()]
        assert "".join(collected) == TEST_RESPONSE


@pytest.mark.asyncio
async def test_agent_streaming_pydanticai_history(test_agent: Agent[None]):
    """Test streaming pydantic-ai history."""
    history = [
        ChatMessage(role="user", content="Previous message"),
        ChatMessage(role="assistant", content="Previous response"),
    ]
    test_agent.conversation.set_history(history)
    stream_ctx = test_agent.run_stream(SIMPLE_PROMPT)
    async with stream_ctx as stream:
        collected = [str(msg) async for msg in stream.stream()]
        result = "".join(collected)
        assert result == TEST_RESPONSE

        # Verify we get the current exchange messages
        new_messages = stream.new_messages()  # type: ignore
        assert len(new_messages) == 2  # Current prompt + response  # noqa: PLR2004

        # Check prompt message
        assert isinstance(new_messages[0], ModelRequest)
        assert isinstance(new_messages[0].parts[0], UserPromptPart)
        assert new_messages[0].parts[0].content == [SIMPLE_PROMPT]

        # Check response message
        assert isinstance(new_messages[1], ModelResponse)
        assert isinstance(new_messages[1].parts[0], TextPart)
        assert new_messages[1].parts[0].content == TEST_RESPONSE


@pytest.mark.asyncio
async def test_agent_concurrent_runs(test_agent: Agent[None]):
    """Test running multiple prompts concurrently."""
    prompts = ["Hello!", "Hi there!", "Good morning!"]
    tasks = [test_agent.run(prompt) for prompt in prompts]
    results = await asyncio.gather(*tasks)
    assert all(r.data == TEST_RESPONSE for r in results)


@pytest.mark.asyncio
async def test_agent_model_override():
    """Test overriding model for specific runs."""
    default_response = "default response"
    override_response = "override response"
    model = TestModel(custom_result_text=default_response)
    async with Agent[None](model=model, name="test-agent") as agent:
        # Run with default model
        result1 = await agent.run(SIMPLE_PROMPT)
        assert result1.data == default_response

        # Run with overridden model
        model2 = TestModel(custom_result_text=override_response)
        result2 = await agent.run(SIMPLE_PROMPT, model=model2)
        assert result2.data == override_response


def test_sync_wrapper(test_agent: Agent[None]):
    """Test synchronous wrapper method."""
    result = test_agent.run_sync(SIMPLE_PROMPT)
    assert result.data == TEST_RESPONSE


@pytest.mark.asyncio
async def test_agent_forwarding():
    """Test message forwarding between agents."""
    async with AgentPool[None]() as pool:
        model = TestModel(custom_result_text="Main response")
        main_agent = await pool.add_agent("main-agent", model=model)
        model = TestModel(custom_result_text="Helper response")
        helper_agent = await pool.add_agent("helper-agent", model=model)

        # Set up forwarding
        main_agent.connect_to(helper_agent)

        # Track messages from both agents
        messages: list[ChatMessage] = []
        main_agent.message_sent.connect(messages.append)
        helper_agent.message_sent.connect(messages.append)

        # Send message and wait for forwarding
        message = "Hello, agent!"

        await main_agent.run(message)
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


if __name__ == "__main__":
    pytest.main([__file__])
