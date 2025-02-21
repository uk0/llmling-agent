"""Global registry for MCP server instances."""

from __future__ import annotations

import asyncio
from datetime import timedelta
import shutil
from typing import Any, Self
import weakref

from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client

from llmling_agent.log import get_logger
from llmling_agent_config.mcp_server import (
    MCPServerConfig,
    SSEMCPServerConfig,
    StdioMCPServerConfig,
)


logger = get_logger(__name__)


class MCPServer:
    """Represents a running MCP server instance."""

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self._initialized = asyncio.Event()
        self._shutdown = asyncio.Event()
        self.name = config.name or "unnamed"
        self.session: ClientSession | None = None

    async def initialize(self):
        """Initialize server if not already done."""
        if self._initialized.is_set():
            return

        try:
            logger.info("Initializing MCP server: %s", self.name)
            match self.config:
                case StdioMCPServerConfig():
                    if not self.config.command or not self.config.args:
                        msg = f"Command and args required for stdio: {self.name}"
                        raise ValueError(msg)  # noqa: TRY301

                    command = shutil.which(self.config.command) or self.config.command
                    server_params = StdioServerParameters(
                        command=command,
                        args=self.config.args,
                        env=self.config.get_env_vars(),
                    )
                    async with stdio_client(server_params) as (read_stream, write_stream):
                        self.session = ClientSession(
                            read_stream,
                            write_stream,
                            read_timeout_seconds=timedelta(seconds=self.config.timeout)
                            if self.config.timeout
                            else None,
                        )
                        await self.session.initialize()

                case SSEMCPServerConfig():
                    if not self.config.url:
                        msg = f"URL required for SSE transport: {self.name}"
                        raise ValueError(msg)  # noqa: TRY301

                    async with sse_client(self.config.url) as (read_stream, write_stream):
                        self.session = ClientSession(
                            read_stream,
                            write_stream,
                            read_timeout_seconds=timedelta(seconds=self.config.timeout)
                            if self.config.timeout
                            else None,
                        )
                        await self.session.initialize()

                case _:
                    msg = f"Unsupported transport: {self.config.transport}"
                    raise ValueError(msg)  # noqa: TRY301

            self._initialized.set()
            logger.info("MCP server initialized: %s", self.name)

        except Exception as e:
            logger.exception("Failed to initialize MCP server: %s", self.name)
            msg = f"Server initialization failed: {e}"
            raise RuntimeError(msg) from e

    async def shutdown(self):
        """Clean shutdown of the server."""
        if not self._initialized.is_set():
            return

        try:
            logger.info("Shutting down MCP server: %s", self.name)
            self._shutdown.set()
            if self.session:
                self.session = None
            self._initialized.clear()
            logger.info("MCP server shut down: %s", self.name)
        except Exception as e:
            logger.exception("Error during server shutdown: %s", self.name)
            msg = f"Server shutdown failed: {e}"
            raise RuntimeError(msg) from e


class ServerRegistry:
    """Global registry for MCP server configurations and instances.

    This is a singleton class that maintains references to all MCP server
    instances. It handles:
    - Server lifecycle management
    - Instance tracking with different configurations
    - Clean shutdown
    """

    _instance: Self | None = None

    def __init__(self):
        """Initialize registry - use get_registry() instead."""
        self._servers: dict[MCPServerConfig, MCPServer] = {}
        self._lock = asyncio.Lock()
        # Use weak references to allow garbage collection
        self._refs = weakref.WeakValueDictionary[MCPServerConfig, MCPServer]()

    @classmethod
    def get_registry(cls) -> Self:
        """Get the global registry instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def get_or_create_server(
        self,
        config: MCPServerConfig,
        **kwargs: Any,
    ) -> MCPServer:
        """Get existing server or create new one.

        Args:
            config: Server configuration
            **kwargs: Additional configuration overrides

        Returns:
            Running server instance

        Raises:
            RuntimeError: If server creation fails
        """
        async with self._lock:
            # Apply overrides
            if kwargs:
                config = config.model_copy(update=kwargs)

            # Return existing server if available
            if config in self._servers:
                return self._servers[config]

            # Create new server
            try:
                server = MCPServer(config)
                await server.initialize()
                self._servers[config] = server
                self._refs[config] = server
            except Exception as e:
                logger.exception("Failed to create MCP server: %s", config.name)
                msg = f"Server creation failed: {e}"
                raise RuntimeError(msg) from e
            else:
                return server

    async def shutdown_server(self, config: MCPServerConfig):
        """Shut down a specific server."""
        async with self._lock:
            if server := self._servers.pop(config, None):
                await server.shutdown()

    async def shutdown_all(self):
        """Clean shutdown of all servers."""
        async with self._lock:
            for server in list(self._servers.values()):
                await server.shutdown()
            self._servers.clear()

    def get_server(self, config: MCPServerConfig) -> MCPServer | None:
        """Get server by config if it exists."""
        return self._servers.get(config)

    def __contains__(self, config: MCPServerConfig) -> bool:
        """Check if server exists for this config."""
        return config in self._servers
