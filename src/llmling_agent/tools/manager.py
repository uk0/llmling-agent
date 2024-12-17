"""Tool management for LLMling agents."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic_ai import Tool


logger = get_logger(__name__)


class ToolManager:
    """Manages tool registration, enabling/disabling and access."""

    def __init__(
        self,
        tools: Sequence[Tool[Any]] = (),
        tool_choice: bool | str | list[str] = True,
    ) -> None:
        """Initialize tool manager.

        Args:
            tools: Initial tools to register
            tool_choice: Control tool usage:
                - True: Allow all tools
                - False: No tools
                - str: Use specific tool
                - list[str]: Allow specific tools
        """
        self._tools = list(tools)
        self._original_tools = list(tools)
        self._disabled_tools: set[str] = set()
        self._tool_choice = tool_choice

    def enable_tool(self, tool_name: str) -> None:
        """Enable a previously disabled tool."""
        self._disabled_tools.discard(tool_name)
        logger.debug("Enabled tool: %s", tool_name)

    def disable_tool(self, tool_name: str) -> None:
        """Disable a tool."""
        self._disabled_tools.add(tool_name)
        logger.debug("Disabled tool: %s", tool_name)

    def is_tool_enabled(self, tool_name: str) -> bool:
        """Check if a tool is currently enabled."""
        return tool_name not in self._disabled_tools

    def list_tools(self) -> dict[str, bool]:
        """Get a mapping of all tools and their enabled status."""
        return {t.name: t.name not in self._disabled_tools for t in self._original_tools}

    def _get_tools(self, tool_names: list[str]) -> list[Tool[Any]]:
        """Get specified tools."""
        return [t for t in self._tools if t.name in tool_names]

    def get_enabled_tools(self) -> list[Tool[Any]]:
        """Get currently enabled tools based on tool_choice setting."""
        match self._tool_choice:
            case False:  # no tools
                return []
            case str() as tool_name:  # specific tool
                return self._get_tools([tool_name])
            case list() as tool_names:  # list of specific tools
                return self._get_tools(tool_names)
            case True:  # auto - return all enabled tools
                return [t for t in self._tools if t.name not in self._disabled_tools]
            case _:
                return []
