"""Tests for AgentPool functionality."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel
import pytest

from llmling_agent.delegation import AgentPool
from llmling_agent.models import AgentConfig, AgentsManifest
from llmling_agent.responses import InlineResponseDefinition, ResponseField


if TYPE_CHECKING:
    from llmling_agent.agent.agent import Agent

MODEL = "openai:gpt-4o-mini"


class ConversationOutput(BaseModel):
    """Test output for conversation flow."""

    message: str
    conversation_index: int


def make_response(prompt: str) -> ConversationOutput:
    """Callback that tracks conversation order."""
    # Track what message we're on in the conversation
    make_response.count = getattr(make_response, "count", 0) + 1  # type: ignore
    return ConversationOutput(
        message=f"Response to: {prompt}",
        conversation_index=make_response.count,  # type: ignore
    )


TEST_CONFIG = f"""\
responses:
  ConversationOutput:
    type: inline
    description: Output with conversation tracking
    fields:
      message:
        type: str
        description: Response message
      conversation_index:
        type: int
        description: Position in conversation

agents:
  test_agent:
    name: Test Agent
    description: Agent for testing conversation flow
    type:
      type: callback
      callback: {__name__}.make_response
    model: test
    result_type: ConversationOutput
    system_prompts:
      - You are a test agent
"""


@pytest.mark.asyncio
async def test_agent_pool_conversation_flow():
    """Test conversation flow maintaining history between messages."""
    manifest = AgentsManifest.from_yaml(TEST_CONFIG)

    async with AgentPool[None](manifest) as pool:
        # Get agent directly for conversation
        agent = pool.get_agent("test_agent")

        # Run multiple prompts in sequence
        responses = []
        prompts = ["Hello!", "How are you?"]

        for prompt in prompts:
            result = await agent.run(prompt)
            responses.append(result)

        # Verify correct number of responses
        assert len(responses) == 2  # noqa: PLR2004

        # Verify conversation order was maintained
        assert responses[0].data.conversation_index == 1
        assert responses[1].data.conversation_index == 2  # noqa: PLR2004

        # Verify message content
        assert responses[0].data.message == "Response to: Hello!"
        assert responses[1].data.message == "Response to: How are you?"

        # Verify agent name
        assert all(r.name == "test_agent" for r in responses)


@pytest.mark.asyncio
async def test_agent_pool_validation():
    """Test AgentPool validation and error handling."""
    fields = {"message": ResponseField(type="str", description="Test message")}
    defn = InlineResponseDefinition(description="Basic test result", fields=fields)
    cfg = AgentConfig(name="Test Agent", model=MODEL, result_type="BasicResult")
    agents = {"test_agent": cfg}
    agent_def = AgentsManifest[Any, Any](responses={"BasicResult": defn}, agents=agents)

    # Test initialization with non-existent agent
    with pytest.raises(ValueError, match="Unknown agents"):
        AgentPool(agent_def, agents_to_load=["nonexistent"])

    # Test getting non-existent agent
    async with AgentPool[None](agent_def) as pool:
        with pytest.raises(KeyError, match="nonexistent"):
            pool.get_agent("nonexistent")


@pytest.mark.asyncio
async def test_agent_pool_team_errors(test_model):
    """Test error handling in team tasks."""
    fields = {"message": ResponseField(type="str", description="Test message")}
    defn = InlineResponseDefinition(description="Basic test result", fields=fields)
    cfg = AgentConfig(name="Test Agent", model=test_model, result_type="BasicResult")
    agents = {"test_agent": cfg}
    agent_def = AgentsManifest[Any, Any](responses={"BasicResult": defn}, agents=agents)

    async with AgentPool[None](agent_def, agents_to_load=["test_agent"]) as pool:
        # Test with non-existent team member
        with pytest.raises(KeyError, match="nonexistent"):
            pool.create_group(["test_agent", "nonexistent"])


@pytest.mark.asyncio
async def test_agent_pool_cleanup():
    """Test proper cleanup of agent resources."""
    fields = {"message": ResponseField(type="str", description="Test message")}
    defn = InlineResponseDefinition(description="Basic test result", fields=fields)
    cfg = AgentConfig(name="Test Agent", model=MODEL, result_type="BasicResult")
    agents = {"test_agent": cfg}
    agent_def = AgentsManifest[Any, Any](responses={"BasicResult": defn}, agents=agents)

    # Use context manager to ensure proper cleanup
    async with AgentPool[None](agent_def) as pool:
        # Add some agents
        agent: Agent[Any] = pool.get_agent("test_agent")
        assert "test_agent" in pool.agents

        # Get runtime reference to check cleanup
        runtime = agent.runtime
        assert runtime is not None

        # Test manual cleanup
        await pool.cleanup()
        assert not pool.agents  # Should be empty after cleanup
        # assert runtime._client is None  # Runtime should be shut down

    # Test context manager cleanup
    assert not pool.agents  # Should still be empty after context exit


@pytest.mark.asyncio
async def test_agent_pool_context_cleanup():
    """Test cleanup through context manager."""
    fields = {"message": ResponseField(type="str", description="Test message")}
    defn = InlineResponseDefinition(description="Basic test result", fields=fields)
    cfg = AgentConfig(name="Test Agent", model=MODEL, result_type="BasicResult")
    agents = {"test_agent": cfg}
    agent_def = AgentsManifest[Any, Any](responses={"BasicResult": defn}, agents=agents)

    runtime_ref = None

    async with AgentPool[None](agent_def) as pool:
        agent: Agent[Any] = pool.get_agent("test_agent")
        runtime_ref = agent.runtime
        assert "test_agent" in pool.agents
        assert runtime_ref is not None

    # After context exit
    assert not pool.agents
    # assert runtime_ref._client is None  # Runtime should be shut down
