"""Test configuration and shared fixtures."""

from __future__ import annotations

from typing import Any

from llmling import RuntimeConfig
from pydantic_ai.models.test import TestModel
import pytest
import yamling

from llmling_agent import Agent, AgentConfig, config_resources
from llmling_agent.delegation import AgentPool
from llmling_agent.models.agents import AgentsManifest
from llmling_agent.responses import InlineResponseDefinition, ResponseField


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
    model: openai:gpt-4
    model_settings:
      retries: 3
      result_retries: 2
    result_type: SupportResult
    system_prompts:
      - You are a support agent
      - "Context: {data}"
  researcher:
    name: Research Agent
    model: openai:gpt-4
    result_type: ResearchResult
    system_prompts:
      - You are a researcher
"""


@pytest.fixture
def runtime() -> RuntimeConfig:
    """Provide a RuntimeConfig for testing."""
    return RuntimeConfig.from_file(config_resources.OPEN_BROWSER)


@pytest.fixture
async def simple_agent(runtime: RuntimeConfig) -> Agent[Any]:
    """Provide a basic text agent."""
    return Agent(runtime=runtime, name="test-agent", model="openai:gpt-4o-mini")


@pytest.fixture
def valid_config() -> dict[str, Any]:
    """Fixture providing valid agent configuration."""
    return yamling.load_yaml(VALID_CONFIG)


@pytest.fixture
def basic_agent_config() -> AgentConfig:
    """Create a basic agent configuration for testing."""
    return AgentConfig(
        name="test_agent",
        model="test",
        result_type="BasicResult",
        system_prompts=["You are a helpful test agent."],
    )


@pytest.fixture
def test_model() -> TestModel:
    """Create a TestModel that returns simple text responses."""
    return TestModel(custom_result_text="Test response", call_tools=[])


@pytest.fixture
def basic_response_def() -> dict[str, InlineResponseDefinition]:
    """Create basic response definitions for testing."""
    response = ResponseField(type="str", description="Test message")
    desc = "Basic test result"
    definition = InlineResponseDefinition(description=desc, fields={"message": response})
    return {"BasicResult": definition}


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
    return AgentsManifest[Any](agents={"agent1": agent_1, "agent2": agent_2})


@pytest.fixture
async def pool(manifest):
    """Create test pool with agents."""
    async with AgentPool[None](manifest) as pool:
        yield pool
