"""Tests for agent orchestration."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError
import pytest

from llmling_agent.models import AgentConfig, AgentsManifest
from llmling_agent.runners import AgentOrchestrator, AgentRunConfig


@pytest.mark.asyncio
async def test_orchestrator_with_complex_config(valid_config: dict[str, Any]) -> None:
    """Test orchestrator with full configuration including multiple agents."""
    agent_def = AgentsManifest.model_validate(valid_config)
    names = ["support", "researcher"]
    config = AgentRunConfig(agent_names=names, prompts=["Hello!"])

    orchestrator = AgentOrchestrator(agent_def, config)
    results = await orchestrator.run()

    assert isinstance(results, dict)
    assert "support" in results
    assert "researcher" in results

    # Test support agent results
    support_results = results["support"]
    assert len(support_results) == 1
    assert isinstance(support_results[0].data, str)

    # Test researcher agent results
    researcher_results = results["researcher"]
    assert len(researcher_results) == 1
    assert isinstance(researcher_results[0].data, str)


@pytest.mark.asyncio
async def test_orchestrator_missing_response(basic_agent_config: AgentConfig) -> None:
    """Test orchestrator with missing response definition."""
    update = {"result_type": "NonExistentResponse"}
    agents = {"test": basic_agent_config.model_copy(update=update)}
    with pytest.raises(ValidationError, match="NonExistentResponse"):
        AgentsManifest(responses={}, agents=agents)
