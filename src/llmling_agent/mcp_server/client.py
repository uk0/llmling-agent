"""FastMCP-based client implementation for LLMling agent."""

from __future__ import annotations

import contextlib
import inspect
import logging
from typing import TYPE_CHECKING, Any, Self

from mcp.types import TextContent

from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from types import TracebackType
    from typing import Any

    import fastmcp
    from fastmcp.client.elicitation import ElicitationHandler, ElicitResult
    from fastmcp.client.logging import LogMessage
    from fastmcp.client.messages import MessageHandler, MessageHandlerT
    from fastmcp.client.sampling import ClientSamplingHandler
    import mcp
    from mcp.client.session import RequestContext
    from mcp.types import (
        CreateMessageRequestParams,
        ElicitRequestParams,
        Prompt as MCPPrompt,
        Resource as MCPResource,
        SamplingMessage,
        Tool as MCPTool,
    )

    from llmling_agent.mcp_server.progress import ProgressHandler
    from llmling_agent.tools.base import Tool
    from llmling_agent_config.mcp_server import MCPServerConfig

logger = get_logger(__name__)

LEVEL_MAP = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "notice": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
    "alert": logging.CRITICAL,
    "emergency": logging.CRITICAL,
}


def mcp_tool_to_fn_schema(tool: MCPTool) -> dict[str, Any]:
    """Convert MCP tool to OpenAI function schema format."""
    return {
        "name": tool.name,
        "description": tool.description or "",
        "parameters": tool.inputSchema or {"type": "object", "properties": {}},
    }


class MCPClient:
    """FastMCP-based client for communicating with MCP servers."""

    def __init__(
        self,
        elicitation_callback: ElicitationHandler | None = None,
        sampling_callback: ClientSamplingHandler | None = None,
        progress_handler: ProgressHandler | None = None,
        message_handler: MessageHandlerT | MessageHandler | None = None,
        accessible_roots: list[str] | None = None,
    ):
        self._elicitation_callback = elicitation_callback
        self._sampling_callback = sampling_callback
        self._progress_handler = progress_handler
        # Store message handler or mark for lazy creation
        self._message_handler = message_handler
        self._use_default_message_handler = message_handler is None
        self._accessible_roots = accessible_roots or []
        self._client: fastmcp.Client | None = None
        self._available_tools: list[MCPTool] = []
        self._connected = False

        # Track current tool execution context for progress notifications
        self._current_tool_name: str | None = None
        self._current_tool_call_id: str | None = None
        self._current_tool_input: dict | None = None

    async def __aenter__(self) -> Self:
        """Enter context manager."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ):
        """Exit context manager and cleanup."""
        await self.cleanup()

    async def cleanup(self):
        """Clean up resources."""
        if self._client:
            try:
                await self._client.__aexit__(None, None, None)
            except Exception as e:  # noqa: BLE001
                logger.warning("Error during FastMCP client cleanup: %s", e)
            finally:
                self._client = None
                self._connected = False
                self._available_tools = []

    async def _log_handler(self, message: LogMessage) -> None:
        """Handle server log messages."""
        msg = message.data.get("msg", "")
        level = LEVEL_MAP.get(message.level.lower(), logging.INFO)
        logger.log(level, "MCP Server: %s", msg)

    async def _progress_handler_impl(
        self, progress: float, total: float | None, message: str | None
    ):
        """Handle progress updates from server."""
        if self._progress_handler:
            await self._progress_handler(
                "",  # tool_name
                "",  # tool_call_id
                {},  # tool_input
                progress,
                total,
                message,
            )

    async def _elicitation_handler_impl(
        self,
        message: str,
        response_type: type,
        params: ElicitRequestParams,
        context: RequestContext,
    ) -> ElicitResult[dict[str, Any]] | dict[str, Any] | None:
        """Handle elicitation requests from server."""
        if not self._elicitation_callback:
            return None

        try:
            # Direct FastMCP callback - no compatibility layer
            return await self._elicitation_callback(
                message, response_type, params, context
            )
        except Exception:
            logger.exception("Elicitation handler failed")
            from fastmcp.client.elicitation import ElicitResult

            return ElicitResult(action="cancel")

    async def _sampling_handler_impl(
        self,
        messages: list[SamplingMessage],
        params: CreateMessageRequestParams,
        context: RequestContext,
    ) -> str:
        """Handle sampling requests from server."""
        if not self._sampling_callback:
            return "Sampling not supported"

        try:
            result = self._sampling_callback(messages, params, context)
            if inspect.iscoroutine(result):
                result = await result
            return str(result)
        except Exception as e:
            logger.exception("Sampling handler failed")
            return f"Sampling failed: {e}"

    async def connect(self, config: MCPServerConfig):
        """Connect to an MCP server using FastMCP.

        Args:
            config: MCP server configuration object
        """
        import fastmcp

        # Create appropriate client based on transport mode
        client_kwargs: dict[str, Any] = {
            "log_handler": self._log_handler,
            "roots": self._accessible_roots if self._accessible_roots else None,
        }

        # Add optional handlers
        if self._progress_handler:
            client_kwargs["progress_handler"] = self._progress_handler_impl
        if self._elicitation_callback:
            client_kwargs["elicitation_handler"] = self._elicitation_handler_impl
        if self._sampling_callback:
            client_kwargs["sampling_handler"] = self._sampling_handler_impl
        # Create message handler if needed
        if self._use_default_message_handler:
            from llmling_agent.mcp_server.message_handler import MCPMessageHandler

            message_handler: MessageHandlerT | MessageHandler | None = MCPMessageHandler(
                self
            )
        else:
            message_handler = self._message_handler

        if message_handler:
            client_kwargs["message_handler"] = message_handler

        # Remove None values
        client_kwargs = {k: v for k, v in client_kwargs.items() if v is not None}

        try:
            # Lazy import fastmcp to avoid startup overhead
            import fastmcp

            from llmling_agent_config.mcp_server import (
                SSEMCPServerConfig,
                StdioMCPServerConfig,
                StreamableHTTPMCPServerConfig,
            )

            match config:
                case StdioMCPServerConfig(command=command, args=args):
                    # Use StdioTransport directly
                    from fastmcp.client.transports import StdioTransport

                    env = config.get_env_vars()
                    transport = StdioTransport(command=command, args=args, env=env)
                    self._client = fastmcp.Client(transport, **client_kwargs)

                case SSEMCPServerConfig(url=url, auth=auth):
                    # FastMCP auto-detects SSE transport from URL
                    sse_url = (
                        url.rstrip("/") + "/sse" if not url.endswith("/sse") else url
                    )
                    # Add OAuth authentication if enabled
                    if auth.oauth:
                        client_kwargs["auth"] = "oauth"
                        logger.debug("SSE client configured with OAuth authentication")
                    else:
                        logger.debug("SSE client configured without OAuth authentication")
                    self._client = fastmcp.Client(sse_url, **client_kwargs)

                case StreamableHTTPMCPServerConfig(url=url, auth=auth):
                    # FastMCP auto-detects streamable HTTP transport
                    # Add OAuth authentication if enabled
                    if auth.oauth:
                        client_kwargs["auth"] = "oauth"
                        logger.debug(
                            "StreamableHTTP client configured with OAuth authentication"
                        )
                    else:
                        logger.debug(
                            "StreamableHTTP client configured without OAuth authentication"
                        )
                    self._client = fastmcp.Client(url, **client_kwargs)

                case _:
                    msg = f"Unsupported server config type: {type(config)}"
                    raise ValueError(msg)  # noqa: TRY301

            # Connect to server
            await self._client.__aenter__()
            self._connected = True

            # Refresh available tools
            await self._refresh_tools()

        except Exception as e:
            msg = f"Failed to connect to MCP server: {e}"
            logger.exception(msg)
            if self._client:
                with contextlib.suppress(Exception):
                    await self._client.__aexit__(None, None, None)
                self._client = None
            raise RuntimeError(msg) from e

    async def _refresh_tools(self) -> None:
        """Refresh the list of available tools from the server."""
        if not self._client or not self._connected:
            return

        try:
            tools = await self._client.list_tools()
            self._available_tools = tools
            logger.debug("Refreshed %d tools from MCP server", len(tools))
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to refresh tools: %s", e)
            self._available_tools = []

    def get_tools(self) -> list[dict[str, Any]]:
        """Get tools in OpenAI function format."""
        return [
            {"type": "function", "function": mcp_tool_to_fn_schema(tool)}
            for tool in self._available_tools
        ]

    async def list_prompts(self) -> list[MCPPrompt]:
        """Get available prompts from the server."""
        if not self._client or not self._connected:
            msg = "Not connected to MCP server"
            raise RuntimeError(msg)

        try:
            return await self._client.list_prompts()
        except Exception as e:
            msg = f"Failed to list prompts: {e}"
            raise RuntimeError(msg) from e

    async def list_resources(self) -> list[MCPResource]:
        """Get available resources from the server."""
        if not self._client or not self._connected:
            msg = "Not connected to MCP server"
            raise RuntimeError(msg)

        try:
            return await self._client.list_resources()
        except Exception as e:
            msg = f"Failed to list resources: {e}"
            raise RuntimeError(msg) from e

    async def get_prompt(
        self, name: str, arguments: dict[str, str] | None = None
    ) -> mcp.types.GetPromptResult:
        """Get a specific prompt's content."""
        if not self._client or not self._connected:
            msg = "Not connected to MCP server"
            raise RuntimeError(msg)

        try:
            return await self._client.get_prompt_mcp(name, arguments)
        except Exception as e:
            msg = f"Failed to get prompt '{name}': {e}"
            raise RuntimeError(msg) from e

    def convert_tool(self, tool: MCPTool) -> Tool:
        """Create a properly typed callable from MCP tool schema."""
        from schemez.functionschema import FunctionSchema

        from llmling_agent import Tool

        schema = mcp_tool_to_fn_schema(tool)
        fn_schema = FunctionSchema.from_dict(schema)
        sig = fn_schema.to_python_signature()

        async def tool_callable(ctx: Any | None = None, **kwargs: Any) -> str:
            """Dynamically generated MCP tool wrapper."""
            # Filter out None values for optional params
            schema_props = tool.inputSchema.get("properties", {})
            required_props = set(tool.inputSchema.get("required", []))

            filtered_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k in required_props or (k in schema_props and v is not None)
            }

            return await self.call_tool(tool.name, filtered_kwargs)

        # Set proper signature and docstring
        tool_callable.__signature__ = sig  # type: ignore
        tool_callable.__annotations__ = fn_schema.get_annotations()
        tool_callable.__name__ = tool.name
        tool_callable.__doc__ = tool.description or "No description provided."
        meta = {"mcp_tool": tool.name}
        return Tool.from_callable(tool_callable, source="mcp", metadata=meta)

    async def call_tool(
        self,
        name: str,
        arguments: dict | None = None,
        tool_call_id: str | None = None,
    ) -> str:
        """Call an MCP tool."""
        if not self._client or not self._connected:
            msg = "Not connected to MCP server"
            raise RuntimeError(msg)

        # Store context for progress tracking
        self._current_tool_name = name
        self._current_tool_call_id = tool_call_id
        self._current_tool_input = arguments or {}

        try:
            # Use FastMCP's call_tool method
            result = await self._client.call_tool(name, arguments or {})

            # FastMCP returns a CallToolResult with structured data
            # For compatibility, return text content
            if result.content:
                if isinstance(result.content[0], TextContent):
                    return result.content[0].text
                # TODO: proper support.
                return str(result.content)
            if result.data is not None:
                return str(result.data)
        except Exception as e:
            msg = f"MCP tool call failed: {e}"
            raise RuntimeError(msg) from e
        else:
            return "Tool executed successfully"
        finally:
            # Clear context after tool execution
            self._current_tool_name = None
            self._current_tool_call_id = None
            self._current_tool_input = None
