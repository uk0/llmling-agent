"""ACP (Agent Client Protocol) server implementation for llmling-agent.

This module provides the main server class for exposing llmling agents via
the Agent Client Protocol.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Self

from acp import AgentSideConnection
from acp.stdio import stdio_streams
from llmling_agent.log import get_logger
from llmling_agent.models.manifest import AgentsManifest
from llmling_agent_acp.acp_agent import LLMlingACPAgent


if TYPE_CHECKING:
    from pathlib import Path

    from tokonomics.model_discovery import ProviderType
    from tokonomics.model_discovery.model_info import ModelInfo

    from acp import Agent as ACPAgent
    from llmling_agent import Agent, AgentPool
    from llmling_agent_providers.base import UsageLimits

logger = get_logger(__name__)


class ACPServer:
    """ACP (Agent Client Protocol) server for llmling-agent using external library.

    Provides a bridge between llmling-agent's Agent system and the standard ACP
    JSON-RPC protocol using the external acp library for robust communication.

    The actual client communication happens via the AgentSideConnection created
    when run() is called, which communicates with the external process over stdio.
    """

    def __init__(
        self,
        agent_pool: AgentPool[Any],
        *,
        usage_limits: UsageLimits | None = None,
        session_support: bool = True,
        file_access: bool = True,
        terminal_access: bool = True,
        providers: list[ProviderType] | None = None,
        debug_messages: bool = False,
        debug_file: str | None = None,
    ) -> None:
        """Initialize ACP server.

        Args:
            agent_pool: AgentPool containing available agents
            usage_limits: Optional usage limits for model requests and tokens
            session_support: Whether to support session-based operations
            file_access: Whether to support file access operations
            terminal_access: Whether to support terminal access operations
            providers: List of providers to use for model discovery (None = openrouter)
            debug_messages: Whether to enable debug message logging
            debug_file: File path for debug message logging
        """
        self.agent_pool = agent_pool
        self._running = False

        # Server configuration
        self._session_support = session_support
        self._file_access = file_access
        self._terminal_access = terminal_access
        self.usage_limits = usage_limits
        self.providers = providers or ["openrouter"]
        self._debug_messages = debug_messages
        self._debug_file = debug_file
        # Model discovery cache
        self._available_models: list[ModelInfo] = []
        self._models_initialized = False

    @classmethod
    async def from_config(
        cls,
        config_path: str | Path,
        *,
        usage_limits: UsageLimits | None = None,
        session_support: bool = True,
        file_access: bool = True,
        terminal_access: bool = True,
        providers: list[ProviderType] | None = None,
        debug_messages: bool = False,
        debug_file: str | None = None,
    ) -> Self:
        """Create ACP server from existing llmling-agent configuration.

        Args:
            config_path: Path to llmling-agent YAML config file
            usage_limits: Optional usage limits for model requests and tokens
            session_support: Enable session loading support
            file_access: Enable file system access
            terminal_access: Enable terminal access
            providers: List of provider types to use for model discovery
            debug_messages: Enable saving JSON messages to file
            debug_file: Path to debug file

        Returns:
            Configured ACP server instance with agent pool from config
        """
        manifest = AgentsManifest.from_file(config_path)
        server = cls(
            agent_pool=manifest.pool,
            usage_limits=usage_limits,
            session_support=session_support,
            file_access=file_access,
            terminal_access=terminal_access,
            providers=providers,
            debug_messages=debug_messages,
            debug_file=debug_file,
        )
        agent_names = list(server.agent_pool.agents.keys())
        logger.info("Created ACP server with agent pool containing: %s", agent_names)

        return server

    def get_agent(self, name: str) -> Agent[Any]:
        """Get agent by name from the pool."""
        return self.agent_pool.get_agent(name)

    async def run(self) -> None:
        """Run the ACP server."""
        if self._running:
            return
        self._running = True

        try:
            if not self.agent_pool:
                logger.error("No agent pool available - cannot start server")
                msg = "No agent pool available"
                raise RuntimeError(msg)  # noqa: TRY301

            # Initialize models on first run
            await self._initialize_models()

            agent_names = list(self.agent_pool.agents.keys())
            msg = "Starting ACP server with %d agents on stdio: %s"
            logger.info(msg, len(agent_names), agent_names)

            # agent factory function
            def create_acp_agent(connection: AgentSideConnection) -> ACPAgent:
                return LLMlingACPAgent(
                    connection=connection,
                    agent_pool=self.agent_pool,
                    available_models=self._available_models,
                    session_support=self._session_support,
                    file_access=self._file_access,
                    terminal_access=self._terminal_access,
                    usage_limits=self.usage_limits,
                )

            reader, writer = await stdio_streams()
            AgentSideConnection(
                create_acp_agent,
                writer,
                reader,
                debug_messages=self._debug_messages,
                debug_file=self._debug_file,
            )

            logger.info(
                "ACP server started with protocol features: "
                "file_access=%s, terminal_access=%s, session_support=%s",
                self._file_access,
                self._terminal_access,
                self._session_support,
            )

            # Keep the connection alive
            try:
                while self._running:
                    await asyncio.sleep(0.1)
            except KeyboardInterrupt:
                logger.info("ACP server shutdown requested")
            except Exception:
                logger.exception("Connection receive task failed")

        except Exception:
            logger.exception("Error running ACP server")
            raise
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """Shutdown the ACP server and cleanup resources."""
        if not self._running:
            return

        self._running = False
        logger.info("Shutting down ACP server")

        try:
            await self.agent_pool.__aexit__(None, None, None)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to cleanup agent pool: %s", e)

        logger.info("ACP server shutdown complete")

    async def __aenter__(self) -> Self:
        """Async context manager entry."""
        await self.agent_pool.__aenter__()
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Async context manager exit."""
        await self.shutdown()

    async def _initialize_models(self) -> None:
        """Initialize available models using tokonomics model discovery."""
        from tokonomics.model_discovery import get_all_models

        if self._models_initialized:
            return
        try:
            logger.info("Discovering available models...")
            self._available_models = await get_all_models(
                include_deprecated=False, providers=self.providers
            )
            self._models_initialized = True
            logger.info("Discovered %d models", len(self._available_models))
        except Exception:
            logger.exception("Failed to discover models")
            self._available_models = []
        finally:
            self._models_initialized = True
