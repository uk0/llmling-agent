"""Models extending pydantic-ai's Model interface with Pydantic functionality."""

from llmling_agent.pydanticai_models.base import PydanticModel
from llmling_agent.pydanticai_models.multi import MultiModel, RandomMultiModel

__all__ = [
    "MultiModel",
    "PydanticModel",
    "RandomMultiModel",
]
