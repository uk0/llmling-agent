"""Run command for agent execution."""

from __future__ import annotations

import asyncio
import signal

import typer as t

from llmling_agent.delegation.pool import AgentPool
from llmling_agent.log import get_logger


logger = get_logger(__name__)


def watch_command(
    config: str = t.Argument(..., help="Path to agent configuration"),
    log_level: str = t.Option("INFO", help="Logging level"),
):
    """Run agents in event-watching mode."""

    async def run_watch():
        async with AgentPool.open(config) as pool:
            # Set up signal handlers
            stop_event = asyncio.Event()
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, stop_event.set)

            logger.info("Starting event watch mode...")
            logger.info("Active agents: %s", ", ".join(pool.list_agents()))
            logger.info("Press Ctrl+C to stop")

            # Wait until stopped
            await stop_event.wait()

    asyncio.run(run_watch())
