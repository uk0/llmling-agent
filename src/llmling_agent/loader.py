"""Load and validate agent configurations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import yamling

from llmling_agent.log import get_logger
from llmling_agent.models import AgentDefinition


if TYPE_CHECKING:
    import os


logger = get_logger(__name__)


def load_agent_config_file(path: str | os.PathLike[str]) -> AgentDefinition:
    """Load agent configuration from YAML file."""
    try:
        data = yamling.load_yaml_file(path)
        return AgentDefinition.model_validate(data)
    except Exception as exc:
        msg = f"Failed to load agent config from {path}"
        raise ValueError(msg) from exc
