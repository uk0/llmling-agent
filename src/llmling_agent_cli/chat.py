"""Interactive chat command."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from llmling.core.log import get_logger
import typer as t

from llmling_agent.chat_session.output import DefaultOutputWriter
from llmling_agent.log import set_handler_level
from llmling_agent_cli import resolve_agent_config


if TYPE_CHECKING:
    from llmling_agent import LLMlingAgent


logger = get_logger(__name__)

CONFIG_HELP = "Override agent config path"
STREAM_CMD = "--stream/--no-stream"
STREAM_HELP = "Enable streaming mode (default: off)"
FORWARD_HELP = "Forward responses to these agents"


def chat_command(
    agent_name: str = t.Argument(help="Name of agent to chat with"),
    session_id: str | None = t.Option(None, "--session-id", "-s", help="Session id"),
    config: str | None = t.Option(None, "--config", "-c", help=CONFIG_HELP),
    model: str | None = t.Option(None, "--model", "-m", help="Override agent's model"),
    stream: bool = t.Option(True, STREAM_CMD, help=STREAM_HELP),
    forward_to: list[str] = t.Option(None, "--forward-to", "-f", help=FORWARD_HELP),  # noqa: B008
    wait_chain: bool = t.Option(True, "--wait-chain", help="Wait for chain completion"),
    log_level: str = t.Option(
        "WARNING",
        "--log-level",
        "-l",
        help="Log level (DEBUG, INFO, WARNING, ERROR)",
        case_sensitive=False,
    ),
):
    """Start interactive chat session with an agent.

    The agent can forward responses to other agents creating a processing chain.
    Use --forward-to to specify target agents and --wait-chain to control whether
    to wait for the full chain to complete.
    """
    from llmling_agent.delegation import AgentPool
    from llmling_agent_cli.chat_session.session import start_interactive_session

    level = getattr(logging, log_level.upper())
    logging.basicConfig(level=level)

    try:
        # Resolve configuration
        try:
            config_path = resolve_agent_config(config)
        except ValueError as e:
            msg = str(e)
            raise t.BadParameter(msg) from e

        async def run_chat():
            # Create pool with main agent and forwarding targets
            agents_to_load = [agent_name]
            if forward_to:
                agents_to_load.extend(forward_to)

            async with AgentPool.open(
                config_path,
                agents=agents_to_load,
                connect_agents=False,  # We'll handle connections manually
            ) as pool:
                # Get main agent
                agent: LLMlingAgent[Any, str] = pool.get_agent(
                    agent_name,
                    model_override=model,
                    session_id=session_id,
                )

                # Set up forwarding if requested
                if forward_to:
                    for target in forward_to:
                        target_agent = pool.get_agent(target)
                        agent.pass_results_to(target_agent)

                await start_interactive_session(
                    agent,
                    pool=pool,
                    stream=stream,
                    wait_chain=wait_chain,
                )

        show_logs = False
        output = DefaultOutputWriter() if show_logs else None

        with set_handler_level(
            level,
            ["llmling_agent", "llmling"],
            session_handler=output,
        ):
            asyncio.run(run_chat())

    except t.Exit:
        raise
    except KeyboardInterrupt:
        print("\nChat session ended.")
    except Exception as e:
        print(f"Error: {e}")
        raise t.Exit(1) from e


if __name__ == "__main__":
    chat_command()
