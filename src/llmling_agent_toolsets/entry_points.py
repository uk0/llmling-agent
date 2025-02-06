"""Entry point based toolset implementation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from epregistry import EntryPointRegistry
from llmling.core.log import get_logger

from llmling_agent.resource_providers.base import ResourceProvider
from llmling_agent.tools.base import ToolInfo


logger = get_logger(__name__)


class EntryPointTools(ResourceProvider):
    """Provider for entry point based tools."""

    def __init__(self, module: str) -> None:
        super().__init__(name=module)
        self.module = module
        self._tools: list[ToolInfo] | None = None
        self.registry = EntryPointRegistry[Callable[..., Any]]("llmling")

    async def get_tools(self) -> list[ToolInfo]:
        """Get tools from entry points."""
        # Return cached tools if available
        if self._tools is not None:
            return self._tools

        self._tools = []
        entry_point = self.registry.get("tools")
        if not entry_point:
            msg = f"No tools entry point found for {self.module}"
            raise ValueError(msg)

        get_tools = entry_point.load()
        for item in get_tools():
            try:
                self._tools.append(
                    ToolInfo.from_callable(
                        item, source="entry_point", metadata={"module": self.module}
                    )
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to load tool from %s: %s", self.module, e)

        return self._tools
