"""Prompt models for agent configuration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar


if TYPE_CHECKING:
    from llmling_agent.prompts.models import PromptTemplate


class BasePromptProvider(ABC):
    """Base class for external prompt providers."""

    name: ClassVar[str]
    supports_versions: ClassVar[bool] = False
    supports_variables: ClassVar[bool] = False

    @abstractmethod
    async def get_prompt(
        self,
        prompt_id: str,
        version: str | None = None,
        variables: dict[str, Any] | None = None,
    ) -> PromptTemplate:
        """Get a prompt by ID."""

    @abstractmethod
    async def list_prompts(self) -> list[str]:
        """List available prompts."""
