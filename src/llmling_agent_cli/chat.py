"""Interactive chat command."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from llmling.core.log import get_logger
import typer as t

from llmling_agent_cli import resolve_agent_config


logger = get_logger(__name__)

CONFIG_HELP = "Override agent config path"

STREAM_CMD = "--stream/--no-stream"
STREAM_HELP = "Enable streaming mode (default: off)"


def chat_command(
    agent_name: str = t.Argument(help="Name of agent to chat with"),
    config: str | None = t.Option(None, "--config", "-c", help=CONFIG_HELP),
    model: str | None = t.Option(None, "--model", "-m", help="Override agent's model"),
    stream: bool = t.Option(True, STREAM_CMD, help=STREAM_HELP),
    log_level: str = t.Option(
        "WARNING",
        "--log-level",
        "-l",
        help="Log level (DEBUG, INFO, WARNING, ERROR)",
        case_sensitive=False,
    ),
):
    """Start interactive chat session with an agent.

    By default, uses non-streaming mode for better support of structured responses
    and debugging. Use --stream to enable streaming mode for real-time responses.
    """
    from llmling_agent import LLMlingAgent
    from llmling_agent_cli.chat_session.session import start_interactive_session

    level = getattr(logging, log_level.upper())
    logging.basicConfig(level=level)
    logging.getLogger("llmling_agent").setLevel(level)
    logging.getLogger("llmling").setLevel(level)

    try:
        # Resolve configuration
        try:
            config_path = resolve_agent_config(config)
        except ValueError as e:
            msg = str(e)
            raise t.BadParameter(msg) from e

        async def run_chat():
            async with LLMlingAgent[Any, str].open_agent(
                config_path,
                agent_name,
                model=model,  # type: ignore[arg-type]
            ) as agent:
                await start_interactive_session(agent, log_level=level, stream=stream)

        asyncio.run(run_chat())

    except t.Exit:
        raise
    except KeyboardInterrupt:
        print("\nChat session ended.")
    except Exception as e:  # noqa: BLE001
        print(f"Error: {e}")
        raise t.Exit(1)  # noqa: B904


if __name__ == "__main__":
    chat_command()
