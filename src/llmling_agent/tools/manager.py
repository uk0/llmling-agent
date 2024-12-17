"""Tool management for LLMling agents."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from collections.abc import Sequence

    from llmling.tools import LLMCallableTool


logger = get_logger(__name__)


class ToolManager:
    """Manages tool registration, enabling/disabling and access."""

    def __init__(
        self,
        tools: Sequence[LLMCallableTool] = (),
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
        self._tools: dict[str, LLMCallableTool] = {t.name: t for t in tools}
        self._original_tools = list(tools)
        self._disabled_tools: set[str] = set()
        self.tool_choice = tool_choice

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

    def get_tools(
        self,
        state: Literal["all", "enabled", "disabled"] = "all",
        names: list[str] | None = None,
    ) -> list[LLMCallableTool]:
        """Get tool objects based on filters."""
        # First filter by state
        match state:
            case "all":
                tools = list(self._tools.values())
            case "enabled":
                tools = [
                    t
                    for name, t in self._tools.items()
                    if name not in self._disabled_tools
                ]
            case "disabled":
                tools = [
                    t for name, t in self._tools.items() if name in self._disabled_tools
                ]

        # Then filter by names if specified
        if names is not None:
            tools = [t for t in tools if t.name in names]

        return tools

    def get_tool_names(
        self, state: Literal["all", "enabled", "disabled"] = "all"
    ) -> set[str]:
        """Get tool names based on state."""
        match state:
            case "all":
                return set(self._tools)
            case "enabled":
                return set(self._tools) - self._disabled_tools
            case "disabled":
                return self._disabled_tools
