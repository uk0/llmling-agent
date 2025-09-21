"""LiteLLM provider utils."""

from __future__ import annotations

from dataclasses import dataclass

from tokonomics.pydanticai_cost import Usage as TokonomicsUsage

from llmling_agent.common_types import ModelProtocol
from llmling_agent.log import get_logger


logger = get_logger(__name__)


@dataclass
class Usage(TokonomicsUsage):
    """Usage information for a model."""

    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return (self.input_tokens or 0) + (self.output_tokens or 0)


class LiteLLMModel(ModelProtocol):
    """Compatible model class for LiteLLM."""

    def __init__(self, model_name: str):
        self._name = model_name

    @property
    def model_name(self) -> str:
        return self._name.replace(":", "/")
