"""Base resource provider interface."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from llmling import BasePrompt

    from llmling_agent.models.resources import ResourceInfo
    from llmling_agent.tools.base import ToolInfo


class ResourceProvider:
    """Base class for resource providers.

    Provides tools, prompts, and other resources to agents.
    Default implementations return empty lists - override as needed.
    """

    def __init__(self, name: str) -> None:
        """Initialize the resource provider."""
        self.name = name

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"

    async def get_tools(self) -> list[ToolInfo]:
        """Get available tools. Override to provide tools."""
        return []

    async def get_prompts(self) -> list[BasePrompt]:
        """Get available prompts. Override to provide prompts."""
        return []

    async def get_resources(self) -> list[ResourceInfo]:
        """Get available resources. Override to provide resources."""
        return []
