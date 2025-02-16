"""Web interface commands."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import pathlib
from tempfile import NamedTemporaryFile

import typer as t

from llmling_agent import AgentPool
from llmling_agent.log import set_handler_level


THEME_HELP = "UI theme (soft/base/monochrome/glass/default)"
MODEL_HELP = "Model to use (e.g. openai:gpt-4o-mini, gpt-4)"

QUICKSTART_CONFIG = """\
agents:
    quickstart:
        name: quickstart
        model: {model}
"""

LOG_HELP = "Log level (DEBUG, INFO, WARNING, ERROR)"
STREAM_HELP = "Enable streaming mode (default: off)"


def quickstart_command(
    model: str = t.Argument("openai:gpt-4o-mini", help=MODEL_HELP),
    log_level: str = t.Option(
        "WARNING", "--log-level", "-l", help=LOG_HELP, case_sensitive=False
    ),
    stream: bool = t.Option(True, "--stream/--no-stream", help=STREAM_HELP),
):
    """Start an ephemeral chat session with minimal setup."""
    from slashed import DefaultOutputWriter

    level = getattr(logging, log_level.upper())
    logging.basicConfig(level=level)
    from llmling_agent_cli.chat_session.session import start_interactive_session

    try:
        # Create temporary agent config
        with NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as tmp_agent:
            tmp_agent.write(QUICKSTART_CONFIG.format(model=model))
            agent_path = tmp_agent.name

        async def run_chat():
            async with AgentPool[None](agent_path) as pool:
                agent = pool.get_agent("quickstart")
                await start_interactive_session(agent, stream=stream)

        show_logs = False
        output = DefaultOutputWriter() if show_logs else None
        loggers = ["llmling_agent", "llmling"]
        with set_handler_level(level, loggers, session_handler=output):
            asyncio.run(run_chat())

    except KeyboardInterrupt:
        print("\nChat session ended.")
    except Exception as e:  # noqa: BLE001
        print(f"Error: {e}")
        raise t.Exit(1) from None
    finally:
        # Cleanup temporary file
        with contextlib.suppress(Exception):
            pathlib.Path(agent_path).unlink(missing_ok=True)
