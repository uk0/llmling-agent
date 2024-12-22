"""Test configuration and shared fixtures."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from llmling import Config
from llmling.config.models import GlobalSettings, LLMCapabilitiesConfig
from llmling.config.runtime import RuntimeConfig
from pydantic_ai.models.test import TestModel
import pytest
import yamling

from llmling_agent import config_resources
from llmling_agent.agent import LLMlingAgent
from llmling_agent.models import AgentConfig
from llmling_agent.responses import InlineResponseDefinition, ResponseField
from llmling_agent.testing.ui import DummyUI


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator


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
async def test_ui() -> AsyncIterator[DummyUI]:
    """Provide a clean DummyUI instance."""
    ui = DummyUI()
    yield ui
    ui.clear_interactions()


@pytest.fixture
async def error_ui() -> AsyncIterator[DummyUI]:
    """Provide a DummyUI that raises errors."""
    ui = DummyUI(raise_errors=True)
    yield ui
    ui.clear_interactions()


@pytest.fixture
async def streaming_ui() -> AsyncIterator[DummyUI]:
    """Provide a DummyUI with streaming responses."""
    ui = DummyUI(
        message_responses={
            "hello": "Hello World!",
            "test": "Test Response",
        }
    )
    yield ui
    ui.clear_interactions()


@pytest.fixture
def runtime() -> RuntimeConfig:
    """Provide a RuntimeConfig for testing."""
    return RuntimeConfig.from_file(config_resources.OPEN_BROWSER)


@pytest.fixture
async def simple_agent(runtime: RuntimeConfig) -> LLMlingAgent[Any, str]:
    """Provide a basic text agent."""
    from llmling_agent.agent import LLMlingAgent

    return LLMlingAgent(
        runtime=runtime,
        name="test-agent",
        model="openai:gpt-4o-mini",
    )


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
async def no_tool_runtime() -> AsyncGenerator[RuntimeConfig, None]:
    """Create a runtime configuration for testing."""
    caps = LLMCapabilitiesConfig(load_resource=False, get_resources=False)
    global_settings = GlobalSettings(llm_capabilities=caps)
    config = Config(global_settings=global_settings)
    runtime = RuntimeConfig.from_config(config)
    await runtime.__aenter__()
    yield runtime
    await runtime.__aexit__(None, None, None)


@pytest.fixture
def test_agent(no_tool_runtime: RuntimeConfig) -> LLMlingAgent[Any, str]:
    """Create an agent with TestModel for testing."""
    return LLMlingAgent(
        runtime=no_tool_runtime,
        name="test-agent",
        model=TestModel(custom_result_text=TEST_RESPONSE),
    )
