"""Interactive chat command."""

from __future__ import annotations

import asyncio

from llmling.core.log import get_logger
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
import typer as t

from llmling_agent import LLMlingAgent
from llmling_agent.chat_session import ChatSessionManager
from llmling_agent.cli import resolve_agent_config


console = Console()
logger = get_logger(__name__)


async def chat_session(agent: LLMlingAgent[str]) -> None:
    """Run interactive chat session with agent."""
    session_manager = ChatSessionManager()
    chat_session = await session_manager.create_session(agent)

    print(f"\nStarted chat with {agent.name}")
    print("Type 'exit' or press Ctrl+C to end the conversation\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input or user_input.lower() == "exit":
            break

        try:
            console.print("\nAgent:", style="bold blue")
            # Use stream=True for live updates
            with Live("", console=console, vertical_overflow="visible") as live:
                response_parts = []
                response_stream = await chat_session.send_message(user_input, stream=True)
                async for chunk in response_stream:
                    response_parts.append(chunk.content)
                    live.update(Markdown("".join(response_parts)))
                print()  # Empty line for readability

        except Exception as e:  # noqa: BLE001
            console.print(f"\nError: {e}", style="bold red")
            import traceback

            traceback.print_exc()
            continue


def chat_command(
    agent_name: str = t.Argument(help="Name of agent to chat with"),
    config: str | None = t.Option(
        None,
        "--config",
        "-c",
        help="Override agent configuration path",
    ),
    model: str | None = t.Option(
        None,
        "--model",
        "-m",
        help="Override agent's model",
    ),
    debug: bool = t.Option(
        False,
        "--debug",
        "-d",
        help="Enable debug output",
    ),
) -> None:
    """Start interactive chat session with an agent."""
    from llmling_agent.cli.chat_session.session import start_interactive_session

    try:
        # Resolve configuration
        try:
            config_path = resolve_agent_config(config)
        except ValueError as e:
            msg = str(e)
            raise t.BadParameter(msg) from e

        async def run_chat() -> None:
            async with LLMlingAgent[str].open_agent(
                config_path,
                agent_name,
                model=model,
            ) as agent:
                await start_interactive_session(agent, debug=debug)

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
