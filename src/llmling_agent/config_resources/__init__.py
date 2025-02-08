"""Package resources for LLMling configuration."""

from __future__ import annotations

import importlib.resources
from typing import Final

AGENTS_TEMPLATE: Final[str] = str(
    importlib.resources.files("llmling_agent.config_resources") / "agents_template.yml"
)

__all__ = ["AGENTS_TEMPLATE"]
