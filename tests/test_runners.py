from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic_ai.models.test import TestModel
import pytest

from llmling_agent.models import AgentConfig, AgentsManifest
from llmling_agent.runners import AgentOrchestrator, AgentRunConfig, SingleAgentRunner
from llmling_agent.runners.exceptions import AgentNotFoundError, NoPromptsError


if TYPE_CHECKING:
    from llmling.config.runtime import RuntimeConfig

    from llmling_agent.responses import ResponseDefinition


@pytest.mark.asyncio
async def test_single_agent_runner_basic(
    basic_agent_config: AgentConfig,
    basic_response_def: dict[str, ResponseDefinition],
    no_tool_runtime: RuntimeConfig,
) -> None:
    """Test basic SingleAgentRunner functionality."""
    async with SingleAgentRunner[str](
        agent_config=basic_agent_config,
        response_defs=basic_response_def,
    ) as runner:
        # Override the model with TestModel
        runner.agent._pydantic_agent.model = TestModel()

        result = await runner.run("Hello!")
        assert isinstance(result.data, str)
        assert result.data  # should not be empty


@pytest.mark.asyncio
async def test_single_agent_runner_conversation(
    basic_agent_config: AgentConfig,
    basic_response_def: dict[str, ResponseDefinition],
) -> None:
    """Test conversation flow with SingleAgentRunner."""
    async with SingleAgentRunner[str](
        agent_config=basic_agent_config,
        response_defs=basic_response_def,
    ) as runner:
        # Override with TestModel that returns specific responses
        runner.agent._pydantic_agent.model = TestModel(custom_result_text="Test response")

        results = await runner.run_conversation(["Hello!", "How are you?"])
        assert len(results) == 2  # noqa: PLR2004
        assert all(r.data == "Test response" for r in results)


@pytest.mark.asyncio
async def test_orchestrator_single_agent(
    basic_agent_config: AgentConfig,
    basic_response_def: dict[str, ResponseDefinition],
) -> None:
    """Test orchestrator with single agent."""
    agents = {"test_agent": basic_agent_config}
    agent_def = AgentsManifest(responses=basic_response_def, agents=agents)

    config = AgentRunConfig(agent_names=["test_agent"], prompts=["Hello!"])

    orchestrator: AgentOrchestrator[Any] = AgentOrchestrator(agent_def, run_config=config)
    results = await orchestrator.run()

    # For single agent, results should be a list of RunResults
    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0].data, str)


@pytest.mark.asyncio
async def test_orchestrator_multiple_agents(
    basic_agent_config: AgentConfig,
    basic_response_def: dict[str, ResponseDefinition],
    test_model: TestModel,
) -> None:
    """Test orchestrator with multiple agents."""
    test_config = basic_agent_config.model_copy(update={"model": test_model})
    agents = {"agent1": test_config, "agent2": test_config}
    agent_def = AgentsManifest(responses=basic_response_def, agents=agents)

    config = AgentRunConfig(agent_names=["agent1", "agent2"], prompts=["Hello!"])

    orchestrator: AgentOrchestrator[Any] = AgentOrchestrator(agent_def, run_config=config)
    results = await orchestrator.run()

    assert isinstance(results, dict)
    assert len(results) == 2  # noqa: PLR2004
    assert all(
        isinstance(r[0].data, str) and r[0].data == "Test response"
        for r in results.values()
    )


@pytest.mark.asyncio
async def test_orchestrator_validation(
    basic_agent_config: AgentConfig,
    basic_response_def: dict[str, ResponseDefinition],
) -> None:
    """Test orchestrator validation."""
    agents = {"test_agent": basic_agent_config}
    agent_def = AgentsManifest(responses=basic_response_def, agents=agents)

    # Test no prompts first
    config = AgentRunConfig(agent_names=["test_agent"], prompts=[])
    orch: AgentOrchestrator[Any] = AgentOrchestrator(agent_def, config)
    with pytest.raises(NoPromptsError):
        orch.validate()

    # Then test missing agent
    config = AgentRunConfig(agent_names=["nonexistent"], prompts=["Hello!"])
    orch2: AgentOrchestrator[Any] = AgentOrchestrator(agent_def, config)
    with pytest.raises(AgentNotFoundError):
        orch2.validate()
