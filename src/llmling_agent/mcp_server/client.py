"""MCP client integration for LLMling agent."""

from __future__ import annotations

from contextlib import AsyncExitStack
from typing import TYPE_CHECKING

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import EmbeddedResource, ImageContent

from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from mcp.types import Tool

logger = get_logger(__name__)


class MCPClient:
    """MCP client for communicating with MCP servers."""

    def __init__(self):
        self.exit_stack = AsyncExitStack()
        self.session: ClientSession | None = None
        self._available_tools: list[Tool] = []

    async def connect(
        self, command: str, args: list[str], env: dict[str, str] | None = None
    ) -> None:
        """Connect to an MCP server.

        Args:
            command: Command to run (e.g. "python" or "node")
            args: Command arguments (e.g. ["server.py"])
            env: Optional environment variables
        """
        params = StdioServerParameters(command=command, args=args, env=env)

        # Set up connection
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(params))
        stdio, write = stdio_transport
        # Set up connection

        # Create and initialize session
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(stdio, write)
        )
        assert self.session
        await self.session.initialize()

        # Get available tools
        result = await self.session.list_tools()
        self._available_tools = result.tools
        logger.info(
            "Connected to MCP server with tools: %s",
            [t.name for t in self._available_tools],
        )

    def get_tools(self) -> list[dict]:
        """Get tools in OpenAI function format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                },
            }
            for tool in self._available_tools
        ]

    async def call_tool(self, name: str, arguments: dict | None = None) -> str:
        """Call an MCP tool.

        Args:
            name: Name of the tool to call
            arguments: Tool arguments

        Returns:
            Tool result content
        """
        if not self.session:
            msg = "Not connected to MCP server"
            raise RuntimeError(msg)

        result = await self.session.call_tool(name, arguments)
        if isinstance(result.content[0], EmbeddedResource | ImageContent):
            msg = "Tool returned an embedded source"
            raise RuntimeError(msg)  # noqa: TRY004
        return result.content[0].text

    async def cleanup(self) -> None:
        """Clean up resources."""
        await self.exit_stack.aclose()
