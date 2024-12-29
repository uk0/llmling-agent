"""Interactive chat command."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import pathlib
from typing import Any

import typer as t

from llmling_agent.chat_session.output import DefaultOutputWriter
from llmling_agent.log import set_handler_level


MODEL_HELP = "Model to use (e.g. openai:gpt-4o-mini, gpt-4)"


# @cli.command(name="quickstart")
def quickstart_command(
    model: str = t.Argument("openai:gpt-4o-mini", help=MODEL_HELP),
    log_level: str = t.Option(
        "WARNING",
        "--log-level",
        "-l",
        help="Log level (DEBUG, INFO, WARNING, ERROR)",
        case_sensitive=False,
    ),
    stream: bool = t.Option(
        True,
        "--stream/--no-stream",
        help="Enable streaming mode (default: off)",
    ),
):
    """Start an ephemeral chat session with minimal setup."""
    from tempfile import NamedTemporaryFile

    import yamling

    from llmling_agent import LLMlingAgent

    level = getattr(logging, log_level.upper())
    logging.basicConfig(level=level)

    from llmling import Config

    from llmling_agent_cli.chat_session.session import start_interactive_session

    cfg = Config().model_dump(mode="json")
    try:
        # Create temporary runtime config
        with NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as tmp_env:
            yamling.dump_yaml(cfg, stream=tmp_env)
            env_path = tmp_env.name

        # Minimal agent config
        minimal_agent_config = {
            "agents": {
                "quickstart": {
                    "name": "quickstart",
                    "model": model,
                    "environment": env_path,  # Will point to our runtime config
                }
            }
        }

        # Create temporary agent config referencing the runtime config
        with NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as tmp_agent:
            # Update environment path to actual path
            minimal_agent_config["agents"]["quickstart"]["environment"] = env_path
            yamling.dump_yaml(minimal_agent_config, stream=tmp_agent)
            agent_path = tmp_agent.name

        async def run_chat():
            # Use open_agent with our temporary configs
            async with LLMlingAgent[Any, Any].open_agent(
                agent_path,
                "quickstart",
            ) as agent:
                await start_interactive_session(agent, stream=stream)

        show_logs = False
        output = DefaultOutputWriter() if show_logs else None

        with set_handler_level(
            level,
            ["llmling_agent", "llmling"],
            session_handler=output,
        ):
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
