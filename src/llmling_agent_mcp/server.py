"""MCP protocol server implementation."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Literal, Self

from mcp.server import NotificationOptions, Server

from llmling_agent_mcp.handlers import register_handlers
from llmling_agent_mcp.log import get_logger
from llmling_agent_mcp.transports.sse import SSEServer
from llmling_agent_mcp.transports.stdio import StdioServer


if TYPE_CHECKING:
    from collections.abc import Coroutine

    import mcp

    from llmling_agent.models.mcp_server import PoolServerConfig
    from llmling_agent.resource_providers.base import ResourceProvider
    from llmling_agent_mcp.transports.base import TransportBase

logger = get_logger(__name__)

TransportType = Literal["stdio", "sse"]


class LLMLingServer:
    """MCP protocol server implementation."""

    def __init__(
        self,
        provider: ResourceProvider,
        config: PoolServerConfig,
        name: str = "llmling-server",
    ) -> None:
        """Initialize server with resource provider.

        Args:
            provider: Resource provider to expose through MCP
            config: Server configuration
            name: Server name for MCP protocol
        """
        self.name = name
        self.provider = provider
        self.config = config

        # Handle Zed mode if enabled
        if config.zed_mode:
            pass
            # TODO: adapt zed wrapper to work with ResourceProvider
            # prepare_runtime_for_zed(runtime)

        self._subscriptions: defaultdict[str, set[mcp.ServerSession]] = defaultdict(set)
        self._tasks: set[asyncio.Task[Any]] = set()

        # Create MCP server
        self.server = Server(name)
        self.server.notification_options = NotificationOptions(
            prompts_changed=True,
            resources_changed=True,
            tools_changed=True,
        )

        # Create transport
        self.transport = self._create_transport(config)
        self._setup_handlers()

    def _create_transport(self, config: PoolServerConfig) -> TransportBase:
        """Create transport instance based on configuration."""
        match config.transport:
            case "stdio":
                return StdioServer(self.server)
            case "sse":
                return SSEServer(
                    self.server,
                    host=config.host,
                    port=config.port,
                    cors_origins=config.cors_origins,
                )
            case _:
                msg = f"Unknown transport type: {config.transport}"
                raise ValueError(msg)

    def _create_task(self, coro: Coroutine[None, None, Any]) -> asyncio.Task[Any]:
        """Create and track an asyncio task."""
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    def _setup_handlers(self) -> None:
        """Register MCP protocol handlers."""
        register_handlers(self)

    async def start(self, *, raise_exceptions: bool = False) -> None:
        """Start the server."""
        try:
            await self.transport.serve(raise_exceptions=raise_exceptions)
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """Shutdown the server."""
        try:
            await self.transport.shutdown()
            # Cancel all pending tasks
            if self._tasks:
                for task in self._tasks:
                    task.cancel()
                await asyncio.gather(*self._tasks, return_exceptions=True)
        finally:
            self._tasks.clear()

    async def __aenter__(self) -> Self:
        """Async context manager entry."""
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Async context manager exit."""
        await self.shutdown()

    @property
    def current_session(self) -> mcp.ServerSession:
        """Get current session from request context."""
        try:
            return self.server.request_context.session
        except LookupError as exc:
            msg = "No active request context"
            raise RuntimeError(msg) from exc

    @property
    def client_info(self) -> mcp.Implementation | None:
        """Get client info from current session."""
        session = self.current_session
        if not session.client_params:
            return None
        return session.client_params.clientInfo

    async def notify_tool_list_changed(self) -> None:
        """Notify clients about tool list changes."""
        try:
            self._create_task(self.current_session.send_tool_list_changed())
        except RuntimeError:
            logger.debug("No active session for notification")
        except Exception:
            logger.exception("Failed to send tool list change notification")

    async def notify_prompt_list_changed(self) -> None:
        """Notify clients about prompt list changes."""
        try:
            self._create_task(self.current_session.send_prompt_list_changed())
        except RuntimeError:
            logger.debug("No active session for notification")
        except Exception:
            logger.exception("Failed to send prompt list change notification")
