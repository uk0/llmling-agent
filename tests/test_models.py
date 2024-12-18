"""Tests for agent configuration models."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import ValidationError
import pytest
import yamling

from llmling_agent.models import AgentsManifest, SystemPrompt


if TYPE_CHECKING:
    from pathlib import Path


def test_valid_system_prompt():
    """Test valid system prompt configurations."""
    prompts = [
        SystemPrompt(type="text", value="You are a helpful agent"),
        SystemPrompt(type="function", value="mymodule.get_prompt"),
        SystemPrompt(type="template", value="Context: {data}"),
    ]
    for prompt in prompts:
        assert prompt.type in ("text", "function", "template")
        assert prompt.value


def test_invalid_system_prompt():
    """Test invalid system prompt configurations."""
    with pytest.raises(ValidationError):
        SystemPrompt(type="invalid", value="test")  # pyright: ignore

    with pytest.raises(ValidationError):
        # missing required value
        SystemPrompt(type="text")  # type: ignore


def test_valid_agent_definition():
    """Test valid complete agent configuration."""
    config = {
        "responses": {
            "TestResponse": {
                "description": "Test response",
                "type": "inline",
                "fields": {
                    "message": {
                        "type": "str",
                        "description": "A message",
                    },
                    "score": {
                        "type": "int",
                        "constraints": {"ge": 0, "le": 100},
                    },
                },
            },
        },
        "agents": {  # This should be a dict of agents
            "test_agent": {  # Key is the agent ID
                "name": "Test Agent",
                "description": "A test agent",
                "model": "openai:gpt-4",
                "model_settings": {},
                "result_type": "TestResponse",
                "system_prompts": [
                    "You are a test agent",
                ],
            },
        },
    }
    agent_def = AgentsManifest.model_validate(config)
    assert agent_def.responses["TestResponse"].fields["score"].constraints == {  # pyright: ignore
        "ge": 0,
        "le": 100,
    }


def test_missing_referenced_response():
    """Test referencing non-existent response model."""
    config = {
        "responses": {},
        "agent": {
            "name": "Test Agent",
            "model": "openai:gpt-4",
            "result_type": "NonExistentResponse",
            "system_prompts": [],
        },
    }
    with pytest.raises(ValidationError):
        AgentsManifest.model_validate(config)


def test_environment_path_resolution(tmp_path: Path) -> None:
    """Test that environment paths are resolved relative to config file."""
    # Create a mock environment config with valid structure
    caps = {"load_resource": False, "get_resources": False}
    env_config = {"global_settings": {"llm_capabilities": caps}}
    env_file = tmp_path / "env.yml"
    env_file.write_text(yamling.dump_yaml(env_config))

    # Create agent config referencing the environment
    agent_config = {
        "responses": {
            "BasicResult": {
                "description": "Test result",
                "type": "inline",
                "fields": {"message": {"type": "str", "description": "Test message"}},
            }
        },
        "agents": {
            "test_agent": {
                "name": "test",
                "model": "test",
                "result_type": "BasicResult",
                "environment": "env.yml",  # Relative path
            }
        },
    }

    config_file = tmp_path / "agents.yml"
    config_file.write_text(yamling.dump_yaml(agent_config))

    # Load the config and verify path resolution
    agent_def = AgentsManifest.from_file(config_file)
    test_agent = agent_def.agents["test_agent"]

    # The environment path should now be resolved
    config = test_agent.get_config()
    assert config.global_settings.llm_capabilities.load_resource is False
