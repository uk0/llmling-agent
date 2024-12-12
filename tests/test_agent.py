"""Tests for the LLMling agent."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from llmling.config.runtime import RuntimeConfig  # noqa: TC002
from pydantic_ai import RunContext, messages
from pydantic_ai.models.test import TestModel
import pytest

from llmling_agent.agent import LLMlingAgent
from llmling_agent.context import AgentContext  # noqa: TC001


if TYPE_CHECKING:
    from pathlib import Path


SIMPLE_PROMPT = "Hello, how are you?"
TEST_RESPONSE = "I am a test response"


@pytest.mark.asyncio
async def test_simple_agent_run(test_agent: LLMlingAgent[str]) -> None:
    """Test basic agent text response."""
    result = await test_agent.run(SIMPLE_PROMPT)
    assert isinstance(result.data, str)
    assert result.data == TEST_RESPONSE
    assert result.cost() is not None


@pytest.mark.asyncio
async def test_agent_message_history(test_agent: LLMlingAgent[str]) -> None:
    """Test agent with message history."""
    history = [
        messages.UserPrompt(content="Previous message"),
        messages.ModelTextResponse(content="Previous response"),
    ]
    result = await test_agent.run(SIMPLE_PROMPT, message_history=history)
    assert result.data == TEST_RESPONSE
    assert test_agent.last_run_messages
    # Check that history was included
    assert len(test_agent.last_run_messages) >= len(history) + 2  # +2 for new exchange


@pytest.mark.asyncio
async def test_agent_streaming(test_agent: LLMlingAgent[str]) -> None:
    """Test agent streaming response."""
    stream_ctx = test_agent.run_stream(SIMPLE_PROMPT)
    async with await stream_ctx as stream:
        collected = [str(message) async for message in stream.stream()]
        assert "".join(collected) == TEST_RESPONSE


@pytest.mark.asyncio
async def test_agent_streaming_with_history(test_agent: LLMlingAgent[str]) -> None:
    """Test streaming with message history."""
    history = [
        messages.UserPrompt(content="Previous message"),
        messages.ModelTextResponse(content="Previous response"),
    ]

    stream_ctx = test_agent.run_stream(SIMPLE_PROMPT, message_history=history)
    async with await stream_ctx as stream:
        collected = []
        async for message in stream.stream():
            if isinstance(message, messages.ModelTextResponse):
                collected.append(message.content)
            elif hasattr(message, "data"):
                collected.append(str(message.data))
            else:
                collected.append(str(message))

        result = "".join(collected)
        assert result == TEST_RESPONSE

        # Verify we get the current exchange messages
        new_messages = stream.new_messages()
        assert len(new_messages) == 2  # Current prompt + response  # noqa: PLR2004
        assert isinstance(new_messages[0], messages.UserPrompt)
        assert isinstance(
            new_messages[1],
            messages.ModelTextResponse | messages.ModelStructuredResponse,
        )
        assert new_messages[0].content == SIMPLE_PROMPT  # Current prompt
        if isinstance(new_messages[1], messages.ModelTextResponse):
            assert new_messages[1].content == TEST_RESPONSE
        else:
            assert str(new_messages[1].data) == TEST_RESPONSE


@pytest.mark.asyncio
async def test_agent_concurrent_runs(test_agent: LLMlingAgent[str]) -> None:
    """Test running multiple prompts concurrently."""
    prompts = ["Hello!", "Hi there!", "Good morning!"]
    tasks = [test_agent.run(prompt) for prompt in prompts]
    results = await asyncio.gather(*tasks)
    assert all(r.data == TEST_RESPONSE for r in results)


@pytest.mark.asyncio
async def test_agent_model_override(no_tool_runtime: RuntimeConfig) -> None:
    """Test overriding model for specific runs."""
    default_response = "default response"
    override_response = "override response"

    agent: LLMlingAgent[str] = LLMlingAgent(
        runtime=no_tool_runtime,
        name="test-agent",
        model=TestModel(custom_result_text=default_response),
    )

    # Run with default model
    result1 = await agent.run(SIMPLE_PROMPT)
    assert result1.data == default_response

    # Run with overridden model
    result2 = await agent.run(
        SIMPLE_PROMPT,
        model=TestModel(custom_result_text=override_response),
    )
    assert result2.data == override_response


@pytest.mark.asyncio
async def test_agent_tool_usage(no_tool_runtime: RuntimeConfig) -> None:
    """Test agent using tools."""
    from pydantic_ai import Tool
    from pydantic_ai.messages import (
        ModelStructuredResponse,
        ModelTextResponse,
        ToolReturn,
        UserPrompt,
    )

    # Create tool before agent initialization
    async def test_tool(ctx: RunContext[AgentContext], message: str = "test") -> str:
        """A test tool."""
        return f"Tool response: {message}"

    tools = [
        Tool(
            test_tool,
            takes_ctx=True,
            name="test_tool",
            description="A test tool.",
        )
    ]

    agent: LLMlingAgent[str] = LLMlingAgent(
        runtime=no_tool_runtime,
        name="test-agent",
        model=TestModel(
            custom_result_text=TEST_RESPONSE,
            call_tools=["test_tool"],
        ),
        tools=tools,  # Pass tools during initialization
    )

    result = await agent.run("Use the test tool")
    assert result.data == TEST_RESPONSE

    # Check message sequence
    messages = result.new_messages()
    assert (
        len(messages) == 4  # noqa: PLR2004
    )  # user prompt -> structured response -> tool return -> final response

    # Check specific message types
    assert isinstance(messages[0], UserPrompt)
    assert isinstance(messages[1], ModelStructuredResponse)
    assert isinstance(messages[2], ToolReturn)
    assert isinstance(messages[3], ModelTextResponse)

    # Verify tool call details
    tool_call = messages[1].calls[0]  # First (and only) tool call
    assert tool_call.tool_name == "test_tool"

    # Verify tool return
    tool_return = messages[2]
    assert isinstance(tool_return, ToolReturn)
    assert tool_return.tool_name == "test_tool"
    assert tool_return.content.startswith("Tool response:")

    # Verify final response
    assert messages[3].content == TEST_RESPONSE


def test_sync_wrapper(test_agent: LLMlingAgent[str]) -> None:
    """Test synchronous wrapper method."""
    result = test_agent.run_sync(SIMPLE_PROMPT)
    assert result.data == TEST_RESPONSE


# @pytest.mark.asyncio
async def test_agent_context_manager(tmp_path: Path) -> None:
    """Test using agent as async context manager."""
    from pydantic_ai.messages import ModelTextResponse, UserPrompt
    import yaml

    # Create a minimal config file
    config = {
        "global_settings": {
            "llm_capabilities": {
                "load_resource": False,
                "get_resources": False,
            }
        }
    }

    # Write config to temporary file
    config_path = tmp_path / "test_config.yml"
    config_path.write_text(yaml.dump(config))
    async with LLMlingAgent.open(
        config_path,
        name="test-agent",
        model=TestModel(custom_result_text=TEST_RESPONSE),
    ) as agent:
        result = await agent.run(SIMPLE_PROMPT)
        assert result.data == TEST_RESPONSE

        # Verify we get expected message sequence
        messages = result.new_messages()
        assert len(messages) == 2  # user prompt -> model response  # noqa: PLR2004
        assert isinstance(messages[0], UserPrompt)
        assert isinstance(messages[1], ModelTextResponse)
        assert messages[1].content == TEST_RESPONSE


@pytest.mark.asyncio
async def test_agent_logging(no_tool_runtime: RuntimeConfig) -> None:
    """Test agent logging functionality."""
    # Test with logging enabled
    agent1: LLMlingAgent[str] = LLMlingAgent(
        runtime=no_tool_runtime,
        name="test-agent",
        model=TestModel(custom_result_text=TEST_RESPONSE),
        enable_logging=True,
    )
    result1 = await agent1.run(SIMPLE_PROMPT)
    assert result1.data == TEST_RESPONSE

    # Test with logging disabled
    agent2: LLMlingAgent[str] = LLMlingAgent(
        runtime=no_tool_runtime,
        name="test-agent",
        model=TestModel(custom_result_text=TEST_RESPONSE),
        enable_logging=False,
    )
    result2 = await agent2.run(SIMPLE_PROMPT)
    assert result2.data == TEST_RESPONSE
