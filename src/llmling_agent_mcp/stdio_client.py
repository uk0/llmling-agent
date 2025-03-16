"""MCP protocol client implementations."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from pydantic import AnyUrl

from llmling_agent_mcp.log import get_logger


if TYPE_CHECKING:
    from collections.abc import Sequence

    from mcp.types import Prompt, Resource, Tool

logger = get_logger(__name__)


@dataclass
class StdioClientConfig:
    """Configuration for stdio-based MCP client."""

    server_command: list[str]
    protocol_version: str = "0.1"
    client_name: str = "mcp-client"
    client_version: str = "1.0"
    timeout: float = 30.0


class MCPClientError(Exception):
    """Base exception for MCP client errors."""


class McpConnectionError(MCPClientError):
    """Raised when connection to server fails."""


class ToolError(MCPClientError):
    """Raised when tool execution fails."""


class StdioMCPClient:
    """High-level MCP protocol client using stdio transport."""

    def __init__(self, config: StdioClientConfig):
        """Initialize client."""
        self.config = config
        self._server_params = StdioServerParameters(
            command=config.server_command[0],
            args=config.server_command[1:],
        )
        self._streams_ctx: Any | None = None
        self._session: ClientSession | None = None

    @classmethod
    async def create(cls, config: StdioClientConfig) -> StdioMCPClient:
        """Create and start a new client instance."""
        client = cls(config)
        try:
            await client.start()
        except Exception as exc:
            await client.close()
            msg = "Failed to create client"
            raise McpConnectionError(msg) from exc
        else:
            return client

    async def start(self):
        """Start client and perform handshake."""
        try:
            # Get streams via stdio_client context manager
            self._streams_ctx = stdio_client(self._server_params)
            assert self._streams_ctx
            read_stream, write_stream = await self._streams_ctx.__aenter__()

            # Create session and initialize
            self._session = ClientSession(read_stream, write_stream)
            await self._session.__aenter__()
            await self._session.initialize()
            msg = "Connected to MCP server (protocol version %s)"
            logger.info(msg, self.config.protocol_version)

        except Exception as exc:
            await self.close()
            msg = "Failed to start MCP client"
            raise McpConnectionError(msg) from exc

    async def close(self):
        """Close client connection."""
        if self._session:
            await self._session.__aexit__(None, None, None)
            self._session = None
        if self._streams_ctx:
            await self._streams_ctx.__aexit__(None, None, None)
            self._streams_ctx = None

    @property
    def session(self) -> ClientSession:
        """Get active session."""
        if not self._session:
            msg = "Not connected to server"
            raise RuntimeError(msg)
        return self._session

    async def list_tools(self) -> Sequence[Tool]:
        """List available tools."""
        try:
            result = await self.session.list_tools()
        except Exception as exc:
            msg = "Failed to list tools"
            raise ToolError(msg) from exc
        else:
            return result.tools

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> Any:
        """Call a tool with given arguments.

        Args:
            name: Name of the tool to call
            arguments: Tool arguments
            timeout: Optional timeout in seconds

        Returns:
            Tool execution result

        Raises:
            ToolError: If tool execution fails
        """
        timeout = timeout or self.config.timeout
        try:
            async with asyncio.timeout(timeout):
                result = await self.session.call_tool(name, arguments)
        except TimeoutError as exc:
            msg = f"Tool execution timed out after {timeout}s"
            raise ToolError(msg) from exc
        except Exception as exc:
            msg = f"Tool execution failed: {exc}"
            raise ToolError(msg) from exc
        else:
            return result.content

    async def list_resources(self) -> Sequence[Resource]:
        """List available resources."""
        try:
            result = await self.session.list_resources()
        except Exception as exc:
            msg = "Failed to list resources"
            raise MCPClientError(msg) from exc
        else:
            return result.resources

    async def list_prompts(self) -> Sequence[Prompt]:
        """List available prompts."""
        try:
            result = await self.session.list_prompts()
        except Exception as exc:
            msg = "Failed to list prompts"
            raise MCPClientError(msg) from exc
        else:
            return result.prompts

    async def subscribe_resource(self, uri: str | AnyUrl):
        """Subscribe to resource updates."""
        try:
            if isinstance(uri, str):
                uri = AnyUrl(uri)
            await self.session.subscribe_resource(uri)
        except Exception as exc:
            msg = "Failed to subscribe to resource"
            raise MCPClientError(msg) from exc

    async def unsubscribe_resource(self, uri: str | AnyUrl):
        """Unsubscribe from resource updates."""
        try:
            if isinstance(uri, str):
                uri = AnyUrl(uri)
            await self.session.unsubscribe_resource(uri)
        except Exception as exc:
            msg = "Failed to unsubscribe from resource"
            raise MCPClientError(msg) from exc

    async def __aenter__(self) -> Self:
        """Async context manager entry."""
        if not self._session:
            await self.start()
        return self

    async def __aexit__(self, *exc: object):
        """Async context manager exit."""
        await self.close()


if __name__ == "__main__":

    async def main():
        """Example usage of StdioMCPClient."""
        config = StdioClientConfig(server_command=["python", "-m", "your_server"])

        async with await StdioMCPClient.create(config) as client:
            # List available tools
            tools = await client.list_tools()
            print("Available tools:", tools)

            # Call a tool
            result = await client.call_tool("example_tool", {"arg": "value"})
            print("Tool result:", result)

    asyncio.run(main())
