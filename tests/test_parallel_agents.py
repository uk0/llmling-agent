"""Tests for parallel agent execution."""

from __future__ import annotations

from llmling import Config
import pytest

from llmling_agent.delegation import AgentPool
from llmling_agent.models import AgentConfig, AgentsManifest
from llmling_agent.responses import InlineResponseDefinition, ResponseField


MODEL = "openai:gpt-4o-mini"


@pytest.mark.asyncio
async def test_parallel_agent_execution(test_model):
    """Test multiple agents executing the same prompts in parallel.

    The AgentPool handles parallel execution of agents, allowing comparison
    of how different agents handle the same input.
    """
    # Define a basic response type
    fields = {"message": ResponseField(type="str", description="Test message")}
    defn = InlineResponseDefinition(description="Basic test result", fields=fields)

    # Define two test agents
    a1 = AgentConfig(
        name="First Agent",
        model=test_model,
        result_type="BasicResult",
        system_prompts=["You are the first agent"],
    )
    a2 = AgentConfig(
        name="Second Agent",
        model=test_model,
        result_type="BasicResult",
        system_prompts=["You are the second agent"],
    )

    # Create manifest with both agents
    agents = {"agent1": a1, "agent2": a2}
    agent_def = AgentsManifest(responses={"BasicResult": defn}, agents=agents)

    # Run agents in parallel using AgentPool
    async with AgentPool(agent_def, agents_to_load=["agent1", "agent2"]) as pool:
        # Use team_task to run both agents in parallel
        responses = await pool.team_task(
            prompt="Process this input",
            team=["agent1", "agent2"],
            mode="parallel",
        )

        # Verify each agent processed the prompt
        assert len(responses) == 2  # noqa: PLR2004

        # Create a dict of results for easier verification
        results = {r.agent_name: r for r in responses}

        # Check both agents are present
        assert "agent1" in results
        assert "agent2" in results

        # Verify both agents succeeded
        for response in responses:
            assert response.success
            assert not response.error
            assert response.response == "Test response"  # test_model's response


@pytest.mark.asyncio
async def test_agent_pool_sequential_execution(test_model):
    """Test multiple agents executing in sequence."""
    # Define agents as before
    fields = {"message": ResponseField(type="str", description="Test message")}
    defn = InlineResponseDefinition(description="Basic test result", fields=fields)
    agents = {
        "agent1": AgentConfig(
            name="First Agent",
            model=test_model,
            result_type="BasicResult",
            system_prompts=["You are the first agent"],
        ),
        "agent2": AgentConfig(
            name="Second Agent",
            model=test_model,
            result_type="BasicResult",
            system_prompts=["You are the second agent"],
        ),
    }
    agent_def = AgentsManifest(responses={"BasicResult": defn}, agents=agents)

    async with AgentPool(agent_def, agents_to_load=["agent1", "agent2"]) as pool:
        # Run agents sequentially
        responses = await pool.team_task(
            prompt="Process this input",
            team=["agent1", "agent2"],
            mode="sequential",
        )

        # Verify sequential execution
        assert len(responses) == 2  # noqa: PLR2004
        # Check execution order (if AgentPool preserves order)
        assert [r.agent_name for r in responses] == ["agent1", "agent2"]

        # Verify results
        for response in responses:
            assert response.success
            assert response.response == "Test response"


@pytest.mark.asyncio
async def test_agent_pool_with_environment_override(test_model):
    """Test AgentPool with environment configuration override."""
    # Basic agent setup
    fields = {"message": ResponseField(type="str", description="Test message")}
    defn = InlineResponseDefinition(description="Basic test result", fields=fields)
    cfg = AgentConfig(name="Test Agent", model=test_model, result_type="BasicResult")
    agents = {"test_agent": cfg}
    agent_def = AgentsManifest(responses={"BasicResult": defn}, agents=agents)

    # Create a test environment configuration
    test_env = Config()

    async with AgentPool(agent_def, agents_to_load=["test_agent"]) as pool:
        # Run with environment override
        responses = await pool.team_task(
            prompt="Test prompt",
            team=["test_agent"],
            environment_override=test_env,  # Pass Config instance instead of path
        )

        assert len(responses) == 1
        response = responses[0]
        assert response.success
        assert response.agent_name == "test_agent"
        assert response.response == "Test response"


@pytest.mark.asyncio
async def test_agent_pool_model_override(test_model):
    """Test AgentPool with model override."""
    fields = {"message": ResponseField(type="str", description="Test message")}
    defn = InlineResponseDefinition(description="Basic test result", fields=fields)
    cfg = AgentConfig(name="Test Agent", model=MODEL, result_type="BasicResult")
    agents = {"test_agent": cfg}
    agent_def = AgentsManifest(responses={"BasicResult": defn}, agents=agents)

    async with AgentPool(agent_def, agents_to_load=["test_agent"]) as pool:
        responses = await pool.team_task(
            prompt="Test prompt",
            team=["test_agent"],
            model_override=test_model,  # Override with test model
        )

        assert len(responses) == 1
        response = responses[0]
        assert response.success
        assert response.agent_name == "test_agent"
        assert response.response == "Test response"
