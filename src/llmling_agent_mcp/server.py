"""MCP protocol server implementation."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Self

from fastmcp import FastMCP
from mcp.server.lowlevel.server import NotificationOptions

import llmling_agent
from llmling_agent.utils.tasks import TaskManager
from llmling_agent_mcp.handlers import register_handlers
from llmling_agent_mcp.log import get_logger


if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractAsyncContextManager

    import mcp
    from mcp.server.lowlevel.server import LifespanResultT

    from llmling_agent.resource_providers.base import ResourceProvider
    from llmling_agent_config.pool_server import MCPPoolServerConfig

    LifespanHandler = Callable[
        [FastMCP[LifespanResultT]],
        AbstractAsyncContextManager[LifespanResultT],
    ]

logger = get_logger(__name__)


class LLMLingServer:
    """MCP protocol server implementation."""

    def __init__(
        self,
        provider: ResourceProvider,
        config: MCPPoolServerConfig,
        lifespan: (LifespanHandler | None) = None,
        instructions: str | None = None,
        name: str = "llmling-server",
    ):
        """Initialize server with resource provider.

        Args:
            provider: Resource provider to expose through MCP
            config: Server configuration
            name: Server name for MCP protocol
            lifespan: Lifespan context manager
            instructions: Instructions for Server usage
        """
        super().__init__()
        self.name = name
        self.task_manager = TaskManager()
        self.provider = provider
        self.config = config

        # Handle Zed mode if enabled
        if config.zed_mode:
            pass
            # TODO: adapt zed wrapper to work with ResourceProvider
            # prepare_runtime_for_zed(runtime)

        self._subscriptions: defaultdict[str, set[mcp.ServerSession]] = defaultdict(set)
        self._tasks: set[asyncio.Task[Any]] = set()

        self.fastmcp = FastMCP(
            instructions=instructions,
            lifespan=lifespan,
            version=llmling_agent.__version__,
        )
        self.server = self.fastmcp._mcp_server
        self.server.notification_options = NotificationOptions(
            prompts_changed=True,
            resources_changed=True,
            tools_changed=True,
        )

        self._setup_handlers()

    def _setup_handlers(self):
        """Register MCP protocol handlers."""
        register_handlers(self)

    async def start(self, *, raise_exceptions: bool = False):
        """Start the server."""
        try:
            if self.config.transport == "stdio":
                await self.fastmcp.run_async(transport=self.config.transport)
            else:
                await self.fastmcp.run_async(
                    transport=self.config.transport,
                    host=self.config.host,
                    port=self.config.port,
                )
        finally:
            await self.shutdown()

    async def shutdown(self):
        """Shutdown the server."""
        try:
            if self._tasks:
                for task in self._tasks:
                    task.cancel()
                await asyncio.gather(*self._tasks, return_exceptions=True)
        finally:
            self._tasks.clear()

    async def __aenter__(self) -> Self:
        """Enter async context and start server."""
        try:
            if self.config.transport == "stdio":
                coro = self.fastmcp.run_async(transport=self.config.transport)
            else:
                coro = self.fastmcp.run_async(
                    transport=self.config.transport,
                    host=self.config.host,
                    port=self.config.port,
                )
            self.task_manager.create_task(coro)
        except Exception as e:
            await self.shutdown()
            msg = "Failed to start server"
            logger.exception(msg, exc_info=e)
            raise RuntimeError(msg) from e
        else:
            return self

    async def __aexit__(self, *exc: object):
        """Shutdown the server."""
        await self.shutdown()

    @property
    def current_session(self) -> mcp.ServerSession:
        """Get current session from request context."""
        try:
            return self.server.request_context.session
        except LookupError as exc:
            msg = "No active request context"
            raise RuntimeError(msg) from exc

    async def report_progress(
        self,
        progress: float,
        total: float | None = None,
        message: str | None = None,
        related_request_id: str | None = None,
    ):
        """Report progress for the current operation."""
        progress_token = (
            self.server.request_context.meta.progressToken
            if self.server.request_context.meta
            else None
        )

        if progress_token is None:
            return

        await self.server.request_context.session.send_progress_notification(
            progress_token=progress_token,
            progress=progress,
            total=total,
            message=message,
            related_request_id=related_request_id,
        )

    @property
    def client_info(self) -> mcp.Implementation | None:
        """Get client info from current session."""
        session = self.current_session
        if not session.client_params:
            return None
        return session.client_params.clientInfo

    async def notify_tool_list_changed(self):
        """Notify clients about tool list changes."""
        try:
            self.task_manager.create_task(self.current_session.send_tool_list_changed())
        except RuntimeError:
            logger.debug("No active session for notification")
        except Exception:
            logger.exception("Failed to send tool list change notification")

    async def notify_prompt_list_changed(self):
        """Notify clients about prompt list changes."""
        try:
            self.task_manager.create_task(self.current_session.send_prompt_list_changed())
        except RuntimeError:
            logger.debug("No active session for notification")
        except Exception:
            logger.exception("Failed to send prompt list change notification")
