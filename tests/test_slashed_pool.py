"""Tests for SlashedPool."""

from __future__ import annotations

from typing import Any

from pydantic_ai.models.test import TestModel
import pytest
from slashed import DefaultOutputWriter

from llmling_agent.agent.slashed_pool import MultiAgentResponse, SlashedPool
from llmling_agent.delegation.pool import AgentPool
from llmling_agent.models.agents import AgentConfig, AgentsManifest


@pytest.fixture
async def pool():
    """Create agent pool with two agents."""
    model_1 = TestModel(custom_result_text="Hello from agent1")
    model_2 = TestModel(custom_result_text="Hello from agent2")
    cfg_1 = AgentConfig(
        name="agent1",
        model=model_1,
        description="First agent",
        system_prompts=["You are agent 1"],
    )
    cfg_2 = AgentConfig(
        name="agent2",
        model=model_2,
        description="Second agent",
        system_prompts=["You are agent 2"],
    )
    manifest = AgentsManifest[Any, Any](agents={"agent1": cfg_1, "agent2": cfg_2})
    async with AgentPool.open(manifest) as agent_pool:
        yield agent_pool


@pytest.fixture
def slashed_pool(pool):
    """Create slashed pool with test output."""
    return SlashedPool(pool, output=DefaultOutputWriter())


@pytest.mark.asyncio
async def test_run_single_and_multi(slashed_pool):
    """Test run() with single agent and broadcast."""
    # Single agent using @ syntax
    single_response = await slashed_pool.run("@agent1 hello")
    assert single_response.content == "Hello from agent1"

    # Single agent using parameter
    single_response = await slashed_pool.run("hello", agent="agent2")
    assert single_response.content == "Hello from agent2"

    # Broadcast to all agents
    responses = await slashed_pool.run("hello everyone")
    assert isinstance(responses, MultiAgentResponse)
    assert len(responses.responses) == 2  # noqa: PLR2004
    assert "agent1" in responses.responses
    assert "agent2" in responses.responses


@pytest.mark.asyncio
async def test_run_iter(slashed_pool):
    """Test run_iter() with single agent and broadcast."""
    # Single agent
    responses = [msg async for msg in slashed_pool.run_iter("hello", agent="agent1")]
    assert len(responses) == 1
    assert responses[0].content == "Hello from agent1"

    # Broadcast - collect in order of completion
    responses = [msg async for msg in slashed_pool.run_iter("hello everyone")]
    assert len(responses) == 2  # noqa: PLR2004
    assert all(msg.content.startswith("Hello") for msg in responses)
