"""Interactive chat command."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import pathlib
from typing import Any

from llmling.core.log import get_logger
import typer as t

from llmling_agent import LLMlingAgent


logger = get_logger(__name__)


# @cli.command(name="quickstart")
def quickstart_command(
    model: str = t.Argument(
        "openai:gpt-4o-mini",
        help="Model to use (e.g. openai:gpt-4o-mini, gpt-4)",
    ),
    log_level: str = t.Option(
        "WARNING",
        "--log-level",
        "-l",
        help="Log level (DEBUG, INFO, WARNING, ERROR)",
        case_sensitive=False,
    ),
    stream: bool = t.Option(
        False,  # Default to False
        "--stream/--no-stream",
        help="Enable streaming mode (default: off)",
    ),
) -> None:
    """Start an ephemeral chat session with minimal setup."""
    from tempfile import NamedTemporaryFile

    import yaml

    level = getattr(logging, log_level.upper())
    logging.basicConfig(level=level)
    logging.getLogger("llmling_agent").setLevel(level)
    logging.getLogger("llmling").setLevel(level)

    from llmling import Config

    from llmling_agent.cli.chat_session.session import start_interactive_session

    cfg = Config().model_dump(mode="json")
    try:
        # Create temporary runtime config
        with NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as tmp_env:
            yaml.dump(cfg, tmp_env)
            env_path = tmp_env.name

        # Minimal agent config
        minimal_agent_config = {
            "agents": {
                "quickstart": {
                    "name": "quickstart",
                    "model": model,
                    "role": "assistant",
                    "environment": env_path,  # Will point to our runtime config
                }
            }
        }

        # Create temporary agent config referencing the runtime config
        with NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as tmp_agent:
            # Update environment path to actual path
            minimal_agent_config["agents"]["quickstart"]["environment"] = env_path
            yaml.dump(minimal_agent_config, tmp_agent)
            agent_path = tmp_agent.name

        async def run_chat() -> None:
            # Use open_agent with our temporary configs
            async with LLMlingAgent[Any].open_agent(
                agent_path,
                "quickstart",
            ) as agent:
                await start_interactive_session(agent, log_level=level, stream=stream)

        asyncio.run(run_chat())

    except KeyboardInterrupt:
        print("\nChat session ended.")
    except Exception as e:  # noqa: BLE001
        print(f"Error: {e}")
        raise t.Exit(1)  # noqa: B904
    finally:
        # Cleanup temporary files

        for path in [env_path, agent_path]:
            with contextlib.suppress(Exception):
                pathlib.Path(path).unlink(missing_ok=True)
