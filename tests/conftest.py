"""Test configuration and shared fixtures."""

from __future__ import annotations

from typing import Any

from pydantic_ai.models.test import TestModel
import pytest
import yamling

from llmling_agent import Agent, AgentConfig, AgentPool, AgentsManifest


TEST_RESPONSE = "I am a test response"
VALID_CONFIG = """\
responses:
  SupportResult:
    type: inline
    description: Support agent response
    fields:
      advice:
        type: str
        description: Support advice
      risk:
        type: int
        constraints:
          ge: 0
          le: 100
  ResearchResult:
    type: inline
    description: Research agent response
    fields:
      findings:
        type: str
        description: Research findings

agents:
  support:
    name: Support Agent
    model: openai:gpt-4o-mini
    result_type: SupportResult
    system_prompts:
      - You are a support agent
      - "Context: {data}"
  researcher:
    name: Research Agent
    model: openai:gpt-4o-mini
    result_type: ResearchResult
    system_prompts:
      - You are a researcher
"""


@pytest.fixture
def valid_config() -> dict[str, Any]:
    """Fixture providing valid agent configuration."""
    return yamling.load_yaml(VALID_CONFIG)


@pytest.fixture
def test_agent() -> Agent[None]:
    """Create an agent with TestModel for testing."""
    model = TestModel(custom_result_text=TEST_RESPONSE)
    return Agent(name="test-agent", model=model)


@pytest.fixture
def manifest():
    """Create test manifest with some agents."""
    agent_1 = AgentConfig(name="agent1", model="test")
    agent_2 = AgentConfig(name="agent2", model="test")
    return AgentsManifest(agents={"agent1": agent_1, "agent2": agent_2})


@pytest.fixture
async def pool(manifest):
    """Create test pool with agents."""
    async with AgentPool[None](manifest) as pool:
        yield pool
