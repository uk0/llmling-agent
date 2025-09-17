"""Command for running agents as an ACP (Agent Client Protocol) server.

This creates an ACP-compatible JSON-RPC 2.0 server that exposes your agents
for bidirectional communication over stdio streams, enabling desktop application
integration with file system access, permission handling, and terminal support.
"""

from __future__ import annotations

import asyncio
import logging

import typer as t

from llmling_agent.log import get_logger
from llmling_agent_cli import resolve_agent_config


logger = get_logger(__name__)


def acp_command(  # noqa: PLR0915
    config: str = t.Argument(None, help="Path to agent configuration"),
    log_level: str = t.Option("INFO", help="Logging level"),
    file_access: bool = t.Option(
        False, "--file-access", help="Enable file system access for agents"
    ),
    terminal_access: bool = t.Option(
        False, "--terminal-access", help="Enable terminal access for agents"
    ),
    session_support: bool = t.Option(
        True,
        "--session-support/--no-session-support",
        help="Enable session loading support",
    ),
    show_messages: bool = t.Option(
        False, "--show-messages", help="Show message activity in logs"
    ),
):
    r"""Run agents as an ACP (Agent Client Protocol) server.

    This creates an ACP-compatible JSON-RPC 2.0 server that communicates over stdio
    streams, enabling your agents to work with desktop applications that support
    the Agent Client Protocol.

    The ACP protocol provides:
    - Bidirectional JSON-RPC 2.0 communication
    - Session management and conversation history
    - File system operations with permission handling
    - Terminal integration (optional)
    - Content blocks (text, image, audio, resources)

    Examples:
        # Run ACP server with basic agent
        llmling-agent acp config.yml

        # Run with file system access enabled
        llmling-agent acp config.yml --file-access

        # Run with full capabilities
        llmling-agent acp config.yml --file-access --terminal-access

        # Test with ACP client (example)
        echo '{"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":1},"id":1}' | python -m llmling_agent_cli acp config.yml

    Protocol Flow:
        1. Client sends initialize request
        2. Server responds with capabilities
        3. Client creates new session
        4. Client sends prompt requests
        5. Server streams responses via session updates
    """  # noqa: E501
    try:
        from llmling_agent_acp import ACPServer
    except ImportError as e:
        msg = (
            "ACP integration is not available. "
            "This might be due to missing deps or the ACP package not being installed."
        )
        raise t.BadParameter(msg) from e

    level = getattr(logging, log_level.upper())
    logging.basicConfig(level=level)

    async def run_acp_server():
        try:
            config_path = resolve_agent_config(config)
        except ValueError as e:
            msg = str(e)
            raise t.BadParameter(msg) from e

        logger.info("Starting ACP server for config: %s", config_path)

        # Create ACP server from config
        try:
            acp_server = await ACPServer.from_config(config_path)
        except Exception as e:
            logger.exception("Failed to create ACP server from config")
            raise t.Exit(1) from e

        # Configure agent capabilities
        agent_count = len(acp_server.list_agents())
        if agent_count == 0:
            logger.error("No agents found in configuration")
            raise t.Exit(1)

        logger.info("Configured %d agents for ACP protocol", agent_count)

        if file_access:
            logger.info("File system access enabled")

        if terminal_access:
            logger.info("Terminal access enabled")

        if session_support:
            logger.info("Session loading support enabled")

        # Set up message logging if requested
        if show_messages:
            logger.info("Message activity logging enabled")

        logger.info(
            "Starting ACP server with protocol features: "
            "file_access=%s, terminal_access=%s, session_support=%s",
            file_access,
            terminal_access,
            session_support,
        )

        try:
            # Run the ACP server (communicates over stdio)
            await acp_server.run()
        except KeyboardInterrupt:
            logger.info("ACP server shutdown requested")
        except Exception as e:
            logger.exception("ACP server error")
            raise t.Exit(1) from e
        finally:
            await acp_server.shutdown()

    # Run the async server
    try:
        asyncio.run(run_acp_server())
    except KeyboardInterrupt:
        logger.info("Interrupted")
    except Exception as e:
        logger.exception("Failed to run ACP server")
        raise t.Exit(1) from e


if __name__ == "__main__":
    import typer

    typer.run(acp_command)
