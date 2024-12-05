"""Test configuration and shared fixtures."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llmling.config.runtime import RuntimeConfig
import pytest

from llmling_agent import config_resources


if TYPE_CHECKING:
    from llmling_agent.agent import LLMlingAgent


@pytest.fixture
def runtime() -> RuntimeConfig:
    """Provide a RuntimeConfig for testing."""
    return RuntimeConfig.from_file(config_resources.OPEN_BROWSER)


@pytest.fixture
async def simple_agent(runtime: RuntimeConfig) -> LLMlingAgent[str]:
    """Provide a basic text agent."""
    from llmling_agent.agent import LLMlingAgent

    return LLMlingAgent(
        runtime=runtime,
        name="test-agent",
        model="openai:gpt-3.5-turbo",
    )
