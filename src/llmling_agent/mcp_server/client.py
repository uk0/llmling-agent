"""MCP client integration for LLMling agent."""

from __future__ import annotations

from contextlib import AsyncExitStack, suppress
import shutil
import sys
from typing import TYPE_CHECKING, Any, Self, TextIO

from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from types import TracebackType

    import mcp
    from mcp import ClientSession
    from mcp.types import Tool, Tool as MCPTool

logger = get_logger(__name__)


def mcp_tool_to_fn_schema(tool: MCPTool) -> dict[str, Any]:
    """Convert MCP tool to OpenAI function schema."""
    desc = tool.description or "No description provided"
    return {"name": tool.name, "description": desc, "parameters": tool.inputSchema}


class MCPClient:
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
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ):
        """Restore stdout if redirected and cleanup."""
        try:
            if self._old_stdout:
                sys.stdout = self._old_stdout
            await self.cleanup()
        except RuntimeError as e:
            if "exit cancel scope in a different task" in str(e):
                logger.warning("Ignoring known MCP cleanup issue: Task context mismatch")
            else:
                raise
        except Exception:
            logger.exception("Error during MCP client cleanup")
            raise

    async def cleanup(self):
        """Clean up resources."""
        with suppress(RuntimeError) as cm:
            await self.exit_stack.aclose()

        if cm and cm.error and "exit cancel scope in a different task" in str(cm.error):
            logger.warning("Ignoring known MCP cleanup issue: Task context mismatch")
        elif cm and cm.error:
            raise cm.error

    async def connect(
        self,
        command: str,
        args: list[str],
        env: dict[str, str] | None = None,
        url: str | None = None,
    ):
        """Connect to an MCP server.

        Args:
            command: Command to run (for stdio servers)
            args: Command arguments (for stdio servers)
            env: Optional environment variables
            url: Server URL (for SSE servers)
        """
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        if url:
            # SSE connection - just a placeholder for now
            logger.info("SSE servers not yet implemented")
            self.session = None
            return
        command = shutil.which(command) or command
        # Stdio connection
        params = StdioServerParameters(command=command, args=args, env=env)
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(params))
        stdio, write = stdio_transport
        session = ClientSession(stdio, write)
        self.session = await self.exit_stack.enter_async_context(session)
        assert self.session
        init_result = await self.session.initialize()
        info = init_result.serverInfo
        # Get available tools
        result = await self.session.list_tools()
        self._available_tools = result.tools
        logger.info("Connected to MCP server %s (%s)", info.name, info.version)
        logger.info("Available tools: %s", len(self._available_tools))

    def get_tools(self) -> list[dict]:
        """Get tools in OpenAI function format."""
        return [
            {"type": "function", "function": mcp_tool_to_fn_schema(tool)}
            for tool in self._available_tools
        ]

    async def list_prompts(self) -> mcp.types.ListPromptsResult:
        """Get available prompts from the server."""
        if not self.session:
            msg = "Not connected to MCP server"
            raise RuntimeError(msg)
        return await self.session.list_prompts()

    async def list_resources(self) -> mcp.types.ListResourcesResult:
        """Get available resources from the server."""
        if not self.session:
            msg = "Not connected to MCP server"
            raise RuntimeError(msg)
        return await self.session.list_resources()

    async def get_prompt(self, name: str) -> mcp.types.GetPromptResult:
        """Get a specific prompt's content."""
        if not self.session:
            msg = "Not connected to MCP server"
            raise RuntimeError(msg)
        return await self.session.get_prompt(name)

    def create_tool_callable(self, tool: MCPTool) -> Callable[..., Awaitable[str]]:
        """Create a properly typed callable from MCP tool schema."""
        from py2openai.functionschema import FunctionSchema

        schema = mcp_tool_to_fn_schema(tool)
        fn_schema = FunctionSchema.from_dict(schema)
        sig = fn_schema.to_python_signature()

        async def tool_callable(**kwargs: Any) -> str:
            """Dynamically generated MCP tool wrapper."""
            return await self.call_tool(tool.name, kwargs)

        # Set proper signature and docstring
        tool_callable.__signature__ = sig  # type: ignore
        tool_callable.__annotations__ = fn_schema.get_annotations()
        tool_callable.__name__ = tool.name
        tool_callable.__doc__ = tool.description or "No description provided."
        return tool_callable

    async def call_tool(self, name: str, arguments: dict | None = None) -> str:
        """Call an MCP tool."""
        from mcp.types import EmbeddedResource, ImageContent

        if not self.session:
            msg = "Not connected to MCP server"
            raise RuntimeError(msg)

        try:
            result = await self.session.call_tool(name, arguments or {})
            if isinstance(result.content[0], EmbeddedResource | ImageContent):
                msg = "Tool returned an embedded resource"
                raise TypeError(msg)  # noqa: TRY301
            return result.content[0].text
        except Exception as e:
            msg = f"MCP tool call failed: {e}"
            raise RuntimeError(msg) from e
