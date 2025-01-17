"""Run command for agent execution."""

from __future__ import annotations

import asyncio

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
        async with AgentPool[None](config) as pool:
            # Set up signal handlers
            await pool.run_event_loop()

    asyncio.run(run_watch())
