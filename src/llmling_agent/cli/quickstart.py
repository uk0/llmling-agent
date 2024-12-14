"""Interactive chat command."""

from __future__ import annotations

import asyncio
import contextlib
import pathlib
from typing import Any

from llmling.core.log import get_logger
import typer as t

from llmling_agent import LLMlingAgent


logger = get_logger(__name__)


# @cli.command(name="quickstart")
def quickstart_command(
    model: str = t.Argument(
        "openai:gpt-3.5-turbo",
        help="Model to use (e.g. openai:gpt-3.5-turbo, gpt-4)",
    ),
    debug: bool = t.Option(False, "--debug", "-d", help="Enable debug output"),
) -> None:
    """Start an ephemeral chat session with minimal setup."""
    from tempfile import NamedTemporaryFile

    import yaml

    from llmling_agent.cli.chat_session.session import start_interactive_session

    # Minimal runtime config
    minimal_runtime_config = {
        "version": "1.0",
        "global_settings": {
            "llm_capabilities": {
                "load_resource": False,
                "get_resources": False,
                "install_package": False,
                "register_tool": False,
                "register_code_tool": False,
            }
        },
        "resources": {},
        "tools": {},
    }

    # Minimal agent config
    minimal_agent_config = {
        "agents": {
            "quickstart": {
                "name": "quickstart",
                "model": model,
                "role": "assistant",
                "environment": "env.yml",  # Will point to our runtime config
            }
        }
    }

    try:
        # Create temporary runtime config
        with NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as tmp_env:
            yaml.dump(minimal_runtime_config, tmp_env)
            env_path = tmp_env.name

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
                await start_interactive_session(agent, debug=debug)

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
