"""History management commands."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from llmling.cli.constants import output_format_opt
from llmling.cli.utils import format_output
from rich.console import Console
from rich.markdown import Markdown
from sqlmodel import Session, select
import typer as t


if TYPE_CHECKING:
    from llmling_agent.storage import Conversation, Message


console = Console()
history_cli = t.Typer(
    name="history",
    help="Conversation history management",
    no_args_is_help=True,
)


def format_conversation(
    conversation: Conversation,
    messages: list[Message],
) -> dict[str, Any]:
    """Format a conversation and its messages for display."""
    return {
        "id": conversation.id,
        "agent": conversation.agent_name,
        "start_time": conversation.start_time.isoformat(),
        "messages": [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
                "model": msg.model,
                "token_usage": msg.token_usage,
            }
            for msg in messages
        ],
    }


@history_cli.command(name="show")
def show_history(
    agent_name: str | None = t.Argument(
        None,
        help="Agent name (shows all if not provided)",
    ),
    # Time-based filtering
    since: datetime | None = t.Option(  # noqa: B008
        None,
        "--since",
        "-s",
        help="Show conversations since (YYYY-MM-DD or YYYY-MM-DD HH:MM)",
    ),
    period: str | None = t.Option(
        None,
        "--period",
        "-p",
        help="Show conversations from last period (1h, 2d, 1w, 1m)",
    ),
    # Content filtering
    query: str | None = t.Option(
        None,
        "--query",
        "-q",
        help="Search in message content",
    ),
    model: str | None = t.Option(
        None,
        "--model",
        "-m",
        help="Filter by model used",
    ),
    # Output control
    limit: int = t.Option(10, "--limit", "-n", help="Number of conversations"),
    compact: bool = t.Option(
        False,
        "--compact",
        help="Show only first/last message of conversations",
    ),
    tokens: bool = t.Option(
        False,
        "--tokens",
        "-t",
        help="Include token usage statistics",
    ),
    output_format: str = output_format_opt,
) -> None:
    """Show conversation history with filtering options.

    Examples:
        # Show last 5 conversations
        llmling-agent history show -n 5

        # Show conversations from last 24 hours
        llmling-agent history show --period 24h

        # Show conversations since specific date
        llmling-agent history show --since 2024-01-01

        # Search for specific content
        llmling-agent history show --query "database schema"

        # Show GPT-4 conversations with token usage
        llmling-agent history show --model gpt-4 --tokens

        # Compact view of recent conversations
        llmling-agent history show --period 1d --compact
    """
    from llmling_agent.storage import Conversation, Message, engine

    with Session(engine) as session:
        # Build base query for conversations
        stmt = select(Conversation).order_by(Conversation.start_time.desc())  # type: ignore

        # Apply filters
        if agent_name:
            stmt = stmt.where(Conversation.agent_name == agent_name)

        if since:
            stmt = stmt.where(Conversation.start_time >= since)

        if period:
            try:
                cutoff = datetime.now() - parse_time_period(period)
                stmt = stmt.where(Conversation.start_time >= cutoff)
            except ValueError as e:
                raise t.BadParameter(str(e)) from e

        if limit:
            stmt = stmt.limit(limit)

        conversations = session.exec(stmt).all()
        results = []

        for conv in conversations:
            # Build message query
            msg_query = (
                select(Message)
                .where(Message.conversation_id == conv.id)
                .order_by(Message.timestamp)  # type: ignore
            )

            if query:
                msg_query = msg_query.where(Message.content.contains(query))  # type: ignore

            if model:
                msg_query = msg_query.where(Message.model == model)

            msgs = session.exec(msg_query).all()

            # Skip conversations with no matching messages
            if query and not msgs:
                continue

            # Handle compact mode
            if compact and msgs:
                msgs = [msgs[0], msgs[-1]] if len(msgs) > 1 else msgs

            # Format conversation
            conv_data = format_conversation(conv, msgs)

            # Add token statistics if requested
            if tokens:
                total_tokens = sum(
                    (msg.token_usage or {}).get("total", 0) for msg in msgs
                )
                completion_tokens = sum(
                    (msg.token_usage or {}).get("completion", 0) for msg in msgs
                )
                prompt_tokens = sum(
                    (msg.token_usage or {}).get("prompt", 0) for msg in msgs
                )
                conv_data["token_usage"] = {
                    "total": total_tokens,
                    "completion": completion_tokens,
                    "prompt": prompt_tokens,
                }

            results.append(conv_data)

        if output_format == "text":
            # Pretty print for text format
            for conv in results:
                console.print(f"\n[bold blue]Conversation {conv['id']}[/]")
                console.print(f"Agent: {conv['agent']}, Started: {conv['start_time']}\n")

                if tokens and "token_usage" in conv:
                    usage = conv["token_usage"]
                    console.print(
                        "[dim]"
                        f"Tokens: {usage['total']:,} total "
                        f"({usage['prompt']:,} prompt, "
                        f"{usage['completion']:,} completion)"
                        "[/]"
                    )
                    console.print()

                for msg in conv["messages"]:
                    role_color = "green" if msg["role"] == "assistant" else "yellow"
                    text = f"[{role_color}]{msg['role'].title()}:[/] ({msg['timestamp']})"
                    console.print(text)
                    console.print(Markdown(msg["content"]))
                    if msg.get("model"):
                        text = f"[dim]Model: {msg['model']}[/]"
                        console.print(text, highlight=False)
                    console.print()
        else:
            format_output(results, output_format)


def parse_time_period(period: str) -> timedelta:
    """Parse time period string into timedelta.

    Examples: 1h, 2d, 1w, 1m
    """
    unit = period[-1].lower()
    value = int(period[:-1])
    match unit:
        case "h":
            return timedelta(hours=value)
        case "d":
            return timedelta(days=value)
        case "w":
            return timedelta(weeks=value)
        case "m":
            return timedelta(days=value * 30)
        case "y":
            return timedelta(days=value * 365)
        case _:
            msg = f"Invalid time unit: {unit}"
            raise ValueError(msg)


@history_cli.command(name="stats")
def show_stats(
    agent_name: str | None = t.Argument(
        None,
        help="Agent name (shows all if not provided)",
    ),
    period: str = t.Option(
        "1d",
        "--period",
        "-p",
        help="Time period (1h, 1d, 1w, 1m, 1y)",
    ),
    group_by: str = t.Option(
        "agent",
        "--group-by",
        "-g",
        help="Group by: agent, model, hour, day",
    ),
    output_format: str = output_format_opt,
) -> None:
    """Show usage statistics.

    Examples:
        # Show stats for all agents
        llmling-agent history stats

        # Show daily stats for specific agent
        llmling-agent history stats myagent --group-by day

        # Show model usage for last week
        llmling-agent history stats --period 1w --group-by model
    """
    from llmling_agent.storage import Conversation, Message, engine

    try:
        cutoff = datetime.now() - parse_time_period(period)
    except ValueError as e:
        raise t.BadParameter(str(e)) from e

    with Session(engine) as session:
        # We need to specify the columns we want to select
        query = (
            select(
                Message.model,  # type: ignore
                Conversation.agent_name,  # type: ignore
                Message.timestamp,  # type: ignore
                Message.token_usage,  # type: ignore
            )
            .join(Conversation)
            .where(Message.timestamp > cutoff)
        )

        if agent_name:
            query = query.where(Conversation.agent_name == agent_name)

        # Now we get tuples with known positions
        rows = session.exec(query).all()

        # Process results
        stats: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"total_tokens": 0, "messages": 0, "models": set()}
        )

        for model, agent, timestamp, token_usage in rows:
            match group_by:
                case "agent":
                    key = agent or "unknown"
                case "model":
                    key = model or "unknown"
                case "hour":
                    key = timestamp.strftime("%Y-%m-%d %H:00")
                case "day":
                    key = timestamp.strftime("%Y-%m-%d")
                case _:
                    msg = f"Invalid group_by: {group_by}"
                    raise t.BadParameter(msg)

            entry = stats[key]
            entry["messages"] += 1
            if token_usage:
                entry["total_tokens"] += token_usage.get("total", 0)
            if model:
                entry["models"].add(model)

        # Format for output
        formatted = [
            {
                "name": key,
                "messages": data["messages"],
                "total_tokens": data["total_tokens"],
                "models": sorted(data["models"]),
            }
            for key, data in stats.items()
        ]

        if output_format == "text":
            console.print(f"\n[bold]Usage Statistics ({period})[/]")
            console.print(f"Grouped by: {group_by}\n")

            for entry in formatted:
                console.print(f"[blue]{entry['name']}[/]")
                console.print(f"  Messages: {entry['messages']}")
                console.print(f"  Total tokens: {entry['total_tokens']:,}")
                if entry["models"]:
                    console.print("  Models: " + ", ".join(entry["models"]))
                console.print()
        else:
            format_output(formatted, output_format)
