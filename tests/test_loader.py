"""Tests for agent configuration loading."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest

from llmling_agent.models import AgentDefinition


if TYPE_CHECKING:
    from pathlib import Path


def test_load_valid_config(valid_config: str):
    """Test loading valid configuration."""
    config = AgentDefinition.model_validate(valid_config)
    assert isinstance(config, AgentDefinition)
    assert config.agents["support"].name == "Support Agent"
    assert "SupportResult" in config.responses


def test_load_invalid_file(caplog):
    """Test loading non-existent file."""
    caplog.set_level(logging.CRITICAL)
    with pytest.raises(ValueError):  # noqa: PT011
        AgentDefinition.from_file("nonexistent.yml")


def test_load_invalid_yaml(tmp_path: Path, caplog):
    """Test loading invalid YAML content."""
    caplog.set_level(logging.CRITICAL)
    invalid_file = tmp_path / "invalid.yml"
    invalid_file.write_text("invalid: yaml: content:")

    with pytest.raises(ValueError):  # noqa: PT011
        AgentDefinition.from_file(invalid_file)
