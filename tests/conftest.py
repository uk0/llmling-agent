"""Test configuration and shared fixtures."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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


@pytest.fixture
def valid_config() -> dict[str, Any]:
    """Fixture providing valid agent configuration."""
    return {
        "responses": {
            "SupportResult": {
                "description": "Support agent response",
                "fields": {
                    "advice": {
                        "type": "str",
                        "description": "Support advice",
                    },
                    "risk": {
                        "type": "int",
                        "constraints": {"ge": 0, "le": 100},
                    },
                },
            },
            "ResearchResult": {
                "description": "Research agent response",
                "fields": {
                    "findings": {
                        "type": "str",
                        "description": "Research findings",
                    },
                },
            },
        },
        "agents": {
            "support": {
                "name": "Support Agent",
                "model": "openai:gpt-4",
                "model_settings": {
                    "retries": 3,
                    "result_retries": 2,
                },
                "result_model": "SupportResult",
                "system_prompts": [
                    {"type": "text", "value": "You are a support agent"},
                    {"type": "template", "value": "Context: {data}"},
                ],
            },
            "researcher": {
                "name": "Research Agent",
                "model": "openai:gpt-4",
                "result_model": "ResearchResult",
                "system_prompts": [
                    {"type": "text", "value": "You are a researcher"},
                ],
            },
        },
    }
