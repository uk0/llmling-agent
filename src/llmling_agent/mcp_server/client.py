"""MCP client integration for LLMling agent."""

from __future__ import annotations

from asyncio import Lock
from contextlib import AsyncExitStack, suppress
import logging
from pathlib import Path
import shutil
from typing import TYPE_CHECKING, Any, Self

from pydantic_ai import RunContext  # noqa: TC002

from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from types import TracebackType

    import mcp
    from mcp import ClientSession
    from mcp.client.session import (
        ElicitationFnT,
        MessageHandlerFnT,
        RequestContext,
        SamplingFnT,
    )
    from mcp.shared.session import ProgressFnT, RequestResponder
    from mcp.types import (
        Prompt as MCPPrompt,
        Resource as MCPResource,
        Tool as MCPTool,
    )

    from llmling_agent.mcp_server.progress import ProgressHandler
    from llmling_agent.tools.base import Tool
    from llmling_agent_config.pool_server import TransportType

logger = get_logger(__name__)


def mcp_tool_to_fn_schema(tool: MCPTool) -> dict[str, Any]:
    """Convert MCP tool to OpenAI function schema."""
    desc = tool.description or "No description provided"
    return {"name": tool.name, "description": desc, "parameters": tool.inputSchema}


async def message_handler(
    message: RequestResponder[mcp.ServerRequest, mcp.ClientResult]
    | mcp.ServerNotification
    | Exception,
):
    """Handle MCP server messages."""
    import mcp
    from mcp.shared.session import RequestResponder
    from mcp.types import ClientResult, EmptyResult

    if isinstance(message, Exception):
        raise message
    if isinstance(message, RequestResponder):
        with message as resp:
            await resp.respond(ClientResult(root=EmptyResult()))
            return
    match message.root:
        case mcp.types.CancelledNotification(params=params):
            logger.info("MCP operation cancelled: %s", params)
        case mcp.ProgressNotification(params=params):
            logger.debug("MCP progress: %s", params)
        case mcp.LoggingMessageNotification(params=params):
            # Map MCP log levels to Python logging integer levels
            level_map = {
                "debug": logging.DEBUG,
                "info": logging.INFO,
                "notice": logging.INFO,
                "warning": logging.WARNING,
                "error": logging.ERROR,
                "critical": logging.CRITICAL,
                "alert": logging.CRITICAL,
                "emergency": logging.CRITICAL,
            }

            log_level = level_map[params.level]
            logger_name = f"mcp.{params.logger}" if params.logger else "mcp.server"
            mcp_logger = get_logger(logger_name)
            log_msg = str(params.data) if params.data else "MCP server log message"
            mcp_logger.log(log_level, log_msg)

        case mcp.ResourceUpdatedNotification(uri=uri):
            logger.info("MCP resource updated: %s", uri)
        case mcp.types.ResourceListChangedNotification(params=params):
            logger.info("MCP resource list changed: %s", params)
        case mcp.types.ToolListChangedNotification(params=params):
            logger.info("MCP tool list changed: %s", params)
        case mcp.types.PromptListChangedNotification(params=params):
            logger.info("MCP prompt list changed: %s", params)


class MCPClient:
    """MCP client for communicating with MCP servers."""

    def __init__(
        self,
        transport_mode: TransportType = "stdio",
        elicitation_callback: ElicitationFnT | None = None,
        sampling_callback: SamplingFnT | None = None,
        progress_handler: ProgressHandler | None = None,
        message_handler: MessageHandlerFnT | None = None,
        accessible_roots: list[str] | None = None,
    ):
        self.exit_stack = AsyncExitStack()
        self.session: ClientSession | None = None
        self._available_tools: list[MCPTool] = []
        self._transport_mode = transport_mode
        self._elicitation_callback = elicitation_callback
        self._sampling_callback = sampling_callback
        self._progress_handler = progress_handler
        self._message_handler = message_handler
        self._accessible_roots = accessible_roots or []
        self._enter_lock = Lock()

    async def __aenter__(self) -> Self:
        """Enter context and redirect stdout if in stdio mode."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ):
        """Restore stdout if redirected and cleanup."""
        try:
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

    async def _default_elicitation_callback(
        self,
        context: RequestContext,
        params: mcp.types.ElicitRequestParams,
    ) -> mcp.types.ElicitResult | mcp.types.ErrorData:
        """Default elicitation callback that returns not supported."""
        import mcp

        return mcp.types.ErrorData(
            code=mcp.types.INVALID_REQUEST,
            message="Elicitation not supported",
        )

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
        async with self._enter_lock:
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(params)
            )
            stdio, write = stdio_transport

            # Create a wrapper that matches the expected signature
            async def elicitation_wrapper(context, params):
                if self._elicitation_callback:
                    return await self._elicitation_callback(context, params)
                return await self._default_elicitation_callback(context, params)

            async def sampling_wrapper(
                context: RequestContext,
                params: mcp.types.CreateMessageRequestParams,
            ) -> mcp.types.CreateMessageResult | mcp.types.ErrorData:
                if self._sampling_callback:
                    return await self._sampling_callback(context, params)
                # If no callback provided, let MCP SDK handle with its default
                import mcp

                return mcp.types.ErrorData(
                    code=mcp.types.INVALID_REQUEST,
                    message="Sampling not supported",
                )

            async def list_roots_wrapper(
                context: RequestContext,
            ) -> mcp.types.ListRootsResult | mcp.types.ErrorData:
                """List accessible filesystem roots."""
                import mcp

                roots = []
                for root_path in self._accessible_roots:
                    try:
                        path = Path(root_path).resolve()
                        if path.exists():
                            from pydantic import FileUrl

                            file_url = FileUrl(path.as_uri())
                            roots.append(
                                mcp.types.Root(uri=file_url, name=path.name or str(path))
                            )
                    except (OSError, ValueError):
                        # Skip invalid paths or inaccessible directories
                        continue

                return mcp.types.ListRootsResult(roots=roots)

            session = ClientSession(
                stdio,
                write,
                elicitation_callback=elicitation_wrapper,
                sampling_callback=sampling_wrapper,
                list_roots_callback=list_roots_wrapper,
                message_handler=self._message_handler,
            )
            self.session = await self.exit_stack.enter_async_context(session)
            assert self.session
            init_result = await self.session.initialize()
            info = init_result.serverInfo
            # Get available tools
            result = await self.session.list_tools()
            self._available_tools = result.tools
            logger.info("Connected to MCP server %s (%s)", info.name, info.version)
            logger.info("Available tools: %s", len(self._available_tools))

    def get_tools(self) -> list[dict[str, Any]]:
        """Get tools in OpenAI function format."""
        return [
            {"type": "function", "function": mcp_tool_to_fn_schema(tool)}
            for tool in self._available_tools
        ]

    async def list_prompts(self) -> list[MCPPrompt]:
        """Get available prompts from the server, handling pagination internally."""
        if not self.session:
            msg = "Not connected to MCP server"
            raise RuntimeError(msg)

        all_prompts = []
        cursor = None
        while True:
            result = await self.session.list_prompts(cursor=cursor)
            all_prompts.extend(result.prompts)

            if result.nextCursor is None:
                break
            cursor = result.nextCursor
        return all_prompts

    async def list_resources(self) -> list[MCPResource]:
        """Get available resources from the server, handling pagination internally."""
        if not self.session:
            msg = "Not connected to MCP server"
            raise RuntimeError(msg)

        all_resources = []
        cursor = None

        while True:
            result = await self.session.list_resources(cursor=cursor)
            all_resources.extend(result.resources)

            if result.nextCursor is None:
                break
            cursor = result.nextCursor

        # Return the same structure but with all resources
        return all_resources

    async def get_prompt(
        self, name: str, arguments: dict[str, str] | None = None
    ) -> mcp.types.GetPromptResult:
        """Get a specific prompt's content."""
        if not self.session:
            msg = "Not connected to MCP server"
            raise RuntimeError(msg)
        return await self.session.get_prompt(name, arguments)

    def convert_tool(self, tool: MCPTool) -> Tool:
        """Create a properly typed callable from MCP tool schema."""
        from schemez.functionschema import FunctionSchema

        from llmling_agent import Tool

        schema = mcp_tool_to_fn_schema(tool)
        fn_schema = FunctionSchema.from_dict(schema)
        sig = fn_schema.to_python_signature()

        async def tool_callable(ctx: RunContext | None = None, **kwargs: Any) -> str:
            """Dynamically generated MCP tool wrapper."""
            # Filter out None values for optional params to avoid MCP validation errors
            # Only include parameters that are either required or have non-None values
            schema_props = tool.inputSchema.get("properties", {})
            required_props = set(tool.inputSchema.get("required", []))
            logger.debug("Tool %s: Original kwargs: %s", tool.name, kwargs)
            logger.debug("Tool %s: Schema props: %s", tool.name, schema_props)
            logger.debug("Tool %s: Required props: %s", tool.name, required_props)
            filtered_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k in required_props or (k in schema_props and v is not None)
            }
            logger.debug("Tool %s: Filtered kwargs: %s", tool.name, filtered_kwargs)
            return await self.call_tool(
                tool.name, filtered_kwargs, tool_call_id=ctx.tool_call_id if ctx else None
            )

        # Set proper signature and docstring
        tool_callable.__signature__ = sig  # type: ignore
        tool_callable.__annotations__ = fn_schema.get_annotations()
        tool_callable.__name__ = tool.name
        tool_callable.__doc__ = tool.description or "No description provided."
        meta = {"mcp_tool": tool.name}
        tool_info = Tool.from_callable(tool_callable, source="mcp", metadata=meta)

        # Log the final signature for debugging
        import inspect

        logger.info(
            "Tool %s: Final signature: %s", tool.name, inspect.signature(tool_callable)
        )
        logger.info(
            "Tool %s: Final annotations: %s",
            tool.name,
            getattr(tool_callable, "__annotations__", {}),
        )

        return tool_info

    async def call_tool(
        self,
        name: str,
        arguments: dict | None = None,
        tool_call_id: str | None = None,
    ) -> str:
        """Call an MCP tool with optional ACP progress bridging."""
        from mcp.types import TextContent, TextResourceContents

        if not self.session:
            msg = "Not connected to MCP server"
            raise RuntimeError(msg)

        # Create progress callback if handler and tool_call_id available
        progress_callback = None
        if self._progress_handler and tool_call_id:
            progress_callback = self._create_progress_callback(
                name, tool_call_id, arguments or {}
            )

        try:
            result = await self.session.call_tool(
                name, arguments or {}, progress_callback=progress_callback
            )
            if not isinstance(result.content[0], TextResourceContents | TextContent):
                msg = "Tool returned a non-text response"
                raise TypeError(msg)  # noqa: TRY301
            return result.content[0].text
        except Exception as e:
            msg = f"MCP tool call failed: {e}"
            raise RuntimeError(msg) from e

    def _create_progress_callback(
        self,
        tool_name: str,
        tool_call_id: str,
        tool_input: dict,
    ) -> ProgressFnT:
        """Create progress callback that uses the progress notification handler."""

        async def progress_callback(
            progress: float, total: float | None = None, message: str | None = None
        ) -> None:
            if not self._progress_handler:
                return

            try:
                await self._progress_handler(
                    tool_name,
                    tool_call_id,
                    tool_input,
                    progress,
                    total,
                    message,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("Progress notification handler failed: %s", e)

        return progress_callback
