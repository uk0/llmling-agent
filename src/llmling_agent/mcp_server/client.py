"""MCP client integration for LLMling agent."""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager, AsyncExitStack
import sys
from typing import TYPE_CHECKING, Self, TextIO

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import EmbeddedResource, ImageContent

from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from types import TracebackType

    from mcp.types import Tool

logger = get_logger(__name__)


class MCPClient(AbstractAsyncContextManager["MCPClient"]):
    """MCP client for communicating with MCP servers."""

    def __init__(self, stdio_mode: bool = False):
        self.exit_stack = AsyncExitStack()
        self.session: ClientSession | None = None
        self._available_tools: list[Tool] = []
        self._old_stdout: TextIO | None = None
        self._stdio_mode = stdio_mode

    async def __aenter__(self) -> Self:
        """Enter context and redirect stdout if in stdio mode."""
        try:
            if self._stdio_mode:
                self._old_stdout = sys.stdout
                sys.stdout = sys.stderr
                logger.info("Redirecting stdout for stdio MCP server")
        except Exception as e:
            msg = "Failed to enter MCP client context"
            logger.exception(msg, exc_info=e)
            raise RuntimeError(msg) from e
        else:
            return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Restore stdout if redirected and cleanup."""
        try:
            if self._old_stdout:
                sys.stdout = self._old_stdout
            await self.cleanup()
        except Exception:
            logger.exception("Error during MCP client cleanup")
            raise

    async def connect(
        self,
        command: str,
        args: list[str],
        env: dict[str, str] | None = None,
        url: str | None = None,
    ) -> None:
        """Connect to an MCP server.

        Args:
            command: Command to run (for stdio servers)
            args: Command arguments (for stdio servers)
            env: Optional environment variables
            url: Server URL (for SSE servers)
        """
        if url:
            # SSE connection - just a placeholder for now
            logger.info("SSE servers not yet implemented")
            self.session = None
            return

        # Stdio connection
        params = StdioServerParameters(command=command, args=args, env=env)
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(params))
        stdio, write = stdio_transport
        session = ClientSession(stdio, write)
        self.session = await self.exit_stack.enter_async_context(session)
        assert self.session
        await self.session.initialize()

        # Get available tools
        result = await self.session.list_tools()
        self._available_tools = result.tools
        msg = "Connected to MCP server with tools: %s"
        logger.info(msg, [t.name for t in self._available_tools])

    def get_tools(self) -> list[dict]:
        """Get tools in OpenAI function format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "No description provided",
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
            msg = "Tool returned an embedded resource"
            raise RuntimeError(msg)  # noqa: TRY004
        return result.content[0].text

    async def cleanup(self) -> None:
        """Clean up resources."""
        await self.exit_stack.aclose()
        self._available_tools = []
