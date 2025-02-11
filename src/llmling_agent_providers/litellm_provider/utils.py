from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llmling_agent.common_types import ModelProtocol
from llmling_agent.log import get_logger


logger = get_logger(__name__)


@dataclass
class Usage:
    total_tokens: int | None
    request_tokens: int | None
    response_tokens: int | None


class LiteLLMModel(ModelProtocol):
    """Compatible model class for LiteLLM."""

    def __init__(self, model_name: str):
        self._name = model_name

    @property
    def model_name(self) -> str:
        return self._name.replace(":", "/")


def convert_message_to_chat(message: Any) -> list[dict[str, str]]:
    """Convert message to chat format."""
    # This is a basic implementation - would need to properly handle
    # different message types and parts
    return [{"role": "user", "content": str(message)}]
