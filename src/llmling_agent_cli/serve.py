"""Command for running agents as an MCP server."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import typer as t

from llmling_agent import AgentPool, AgentsManifest, ChatMessage
from llmling_agent.log import get_logger
from llmling_agent_config.mcp_server import PoolServerConfig


logger = get_logger(__name__)


def serve_command(
    config: str = t.Argument(..., help="Path to agent configuration"),
    transport: str = t.Option("stdio", help="Transport type (stdio/sse)"),
    host: str = t.Option("localhost", help="Host to bind server to (sse only)"),
    port: int = t.Option(8000, help="Port to listen on (sse only)"),
    zed_mode: bool = t.Option(False, help="Enable Zed editor compatibility"),
    show_messages: bool = t.Option(
        False, "--show-messages", help="Show message activity"
    ),
    log_level: str = t.Option("INFO", help="Logging level"),
):
    """Run agents as an MCP server.

    This makes agents available as tools to other applications, regardless of
    whether pool_server is configured in the manifest.
    """
    level = getattr(logging, log_level.upper())
    logging.basicConfig(level=level)

    def on_message(message: ChatMessage[Any]):
        print(message.format(style="simple"))

    async def run_server():
        # Create server config that overrides any manifest settings
        # Load manifest
        manifest = AgentsManifest.from_file(config)

        # Override/set server config before creating pool
        manifest.pool_server = PoolServerConfig(
            enabled=True,
            transport=transport,  # type: ignore
            host=host,
            port=port,
            zed_mode=zed_mode,
        )

        # Create pool with modified manifest
        async with AgentPool[None](manifest) as pool:
            # Optionally show messages
            if show_messages:
                for agent in pool.agents.values():
                    agent.message_sent.connect(on_message)

            try:
                await pool.run_event_loop()
            except KeyboardInterrupt:
                logger.info("Server shutdown requested")

    asyncio.run(run_server())
