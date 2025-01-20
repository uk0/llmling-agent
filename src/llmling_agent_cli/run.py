"""Run command for agent execution."""

from __future__ import annotations

import asyncio
import traceback
from typing import TYPE_CHECKING, Any

from llmling.cli.constants import verbose_opt
import typer as t

from llmling_agent.delegation import AgentPool
from llmling_agent_cli import resolve_agent_config


if TYPE_CHECKING:
    from llmling_agent.models.messages import ChatMessage


def run_command(
    agent_name: str = t.Argument(help="Agent name(s) to run (can be comma-separated)"),
    prompts: list[str] = t.Argument(None, help="Additional prompts to send"),  # noqa: B008
    config_path: str = t.Option(None, "-c", "--config", help="Override config path"),
    execution_mode: str = t.Option(
        "parallel",
        "-x",
        "--execution",
        help="Execution mode for multiple agents: parallel/sequential/controlled",
    ),
    show_messages: bool = t.Option(
        True, "--show-messages", help="Show all messages (not just final responses)"
    ),
    detail_level: str = t.Option(
        "simple", "-d", "--detail", help="Output detail level: simple/detailed/markdown"
    ),
    show_metadata: bool = t.Option(False, "--metadata", help="Show message metadata"),
    show_costs: bool = t.Option(False, "--costs", help="Show token usage and costs"),
    model: str = t.Option(None, "--model", "-m", help="Override model"),
    verbose: bool = verbose_opt,
):
    """Run agent(s) with prompts.

    Examples:
        # Single agent
        llmling-agent run myagent "Analyze this"

        # Parallel execution (default)
        llmling-agent run "agent1,agent2,agent3" "Process this"

        # Sequential chain
        llmling-agent run "agent1,agent2,agent3" -x sequential "Process this"

        # Controlled routing (interactive)
        llmling-agent run "agent1,agent2,agent3" -x controlled "Process this"

        # Show all messages
        llmling-agent run "agent1,agent2" --show-messages "Process this"
    """
    try:
        # Resolve configuration path
        try:
            config_path = resolve_agent_config(config_path)
        except ValueError as e:
            error_msg = str(e)
            raise t.BadParameter(error_msg) from e

        async def run():
            async with AgentPool[None](config_path) as pool:

                def on_message(chat_message: ChatMessage[Any]):
                    print(
                        chat_message.format(
                            style=detail_level,  # type: ignore
                            show_metadata=show_metadata,
                            show_costs=show_costs,
                        )
                    )

                # Connect message handlers if showing all messages
                if show_messages:
                    for agent in pool.agents.values():
                        agent.message_sent.connect(on_message)

                agent_names = [name.strip() for name in agent_name.split(",")]
                group = pool.create_team(agent_names, model_override=model)

                for prompt in prompts:
                    match execution_mode:
                        case "parallel":
                            responses = await group.run_parallel(prompt)
                        case "sequential":
                            responses = await group.run_sequential(prompt)
                        case "controlled":
                            responses = await group.run_controlled(prompt)
                        case _:
                            error_msg = f"Invalid execution mode: {execution_mode}"
                            raise t.BadParameter(error_msg)  # noqa: TRY301

                    if not show_messages:
                        messages = [r.message for r in responses]
                        for msg in messages:
                            assert msg
                            print(
                                msg.format(
                                    style=detail_level,  # type: ignore
                                    show_metadata=show_metadata,
                                    show_costs=show_costs,
                                )
                            )

        # Run the async code in the sync command
        asyncio.run(run())

    except t.Exit:
        raise
    except Exception as e:
        t.echo(f"Error: {e}", err=True)
        if verbose:
            t.echo(traceback.format_exc(), err=True)
        raise t.Exit(1) from e
