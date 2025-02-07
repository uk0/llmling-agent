"""Package resources for LLMling configuration."""

from __future__ import annotations

import importlib.resources
from typing import Final

SUMMARIZE_README: Final[str] = str(
    importlib.resources.files("llmling_agent.config_resources") / "summarize_readme.yml"
)
AGENTS_TEMPLATE: Final[str] = str(
    importlib.resources.files("llmling_agent.config_resources") / "agents_template.yml"
)

__all__ = ["AGENTS_TEMPLATE", "SUMMARIZE_README"]
