"""Tests for parallel agent execution."""

from __future__ import annotations

from typing import Any

import pytest

from llmling_agent.models import AgentConfig, AgentsManifest
from llmling_agent.responses import InlineResponseDefinition, ResponseField
from llmling_agent.runners import AgentOrchestrator, AgentRunConfig


@pytest.mark.asyncio
async def test_parallel_agent_execution(test_model):
    """Test multiple agents executing the same prompts in parallel.

    The orchestrator runs each prompt through all agents, allowing comparison
    of how different agents handle the same input.
    """
    fields = {"message": ResponseField(type="str", description="Test message")}
    defn = InlineResponseDefinition(description="Basic test result", fields=fields)
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
    agents = {"agent1": a1, "agent2": a2}
    agent_def = AgentsManifest(responses={"BasicResult": defn}, agents=agents)

    # Run same prompt through multiple agents
    config = AgentRunConfig(
        agent_names=["agent1", "agent2"],
        prompts=["Process this input"],  # Same prompt for all agents
    )

    orchestrator: AgentOrchestrator[Any] = AgentOrchestrator(agent_def, config)
    results = await orchestrator.run()

    # Verify each agent processed the prompt
    assert "agent1" in results
    assert "agent2" in results
    assert len(results["agent1"]) == 1
    assert len(results["agent2"]) == 1

    # Both agents should return test model's response
    for agent_results in results.values():
        assert len(agent_results) == 1
        result = agent_results[0]
        assert result.data == "Test response"
