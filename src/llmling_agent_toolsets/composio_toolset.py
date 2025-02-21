"""Composio based toolset implementation."""

from __future__ import annotations

import os

from llmling.core.log import get_logger

from llmling_agent.resource_providers.base import ResourceProvider
from llmling_agent.tools.base import Tool


logger = get_logger(__name__)


class ComposioTools(ResourceProvider):
    """Provider for composio tools."""

    def __init__(self, entity_id: str, api_key: str | None = None):
        from composio_openai import ComposioToolSet

        super().__init__(name=entity_id)
        self.entity_id = entity_id
        key = api_key or os.environ.get("COMPOSIO_API_KEY")
        self.toolset = ComposioToolSet(entity_id=entity_id, api_key=key)
        self._tools: list[Tool] | None = None

    def _create_tool_handler(self, function_name: str):
        """Create a handler function for a specific tool."""

        def handle_tool_call(**kwargs):
            self.toolset.execute_action(function_name, kwargs)

        handle_tool_call.__name__ = function_name
        return handle_tool_call

    async def get_tools(self) -> list[Tool]:
        """Get tools from entry points."""
        from composio_openai import App

        # Return cached tools if available
        if self._tools is not None:
            return self._tools

        self._tools = []
        for tool_schema in self.toolset.get_tools(apps=[App.GITHUB]):
            function_name = tool_schema["function"]["name"]
            fn = self._create_tool_handler(function_name)
            tool = Tool.from_callable(fn, schema_override=tool_schema["function"])  # type: ignore
            self._tools.append(tool)

        return self._tools


if __name__ == "__main__":
    import asyncio

    async def main():
        from llmling_agent import Agent

        tools = ComposioTools("default")
        agent = Agent[None](model="gpt-4o-mini")
        agent.tools.add_provider(tools)
        result = await agent.run("tell me the tools at your disposal")
        print(result)

    asyncio.run(main())
