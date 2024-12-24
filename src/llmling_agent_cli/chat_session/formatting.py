from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from rich.console import Console
from rich.style import Style


if TYPE_CHECKING:
    from rich.markdown import Markdown

    from llmling_agent.models.messages import ChatMessage, MessageMetadata


class MessageFormatter:
    """Format chat messages for CLI display."""

    LINE_WIDTH: ClassVar[int] = 80
    USER_STYLE: ClassVar[Style] = Style(color="blue")
    ASSISTANT_STYLE: ClassVar[Style] = Style(color="green")
    SYSTEM_STYLE: ClassVar[Style] = Style(color="yellow")
    STATS_STYLE: ClassVar[Style] = Style(dim=True)

    def __init__(self, console: Console | None = None):
        self.console = console or Console()

    def print_message_start(self, message: ChatMessage) -> None:
        """Print message header."""
        self.console.print()  # Space before message
        sender = self._get_sender_name(message)
        line = f"─── {sender} " + "─" * (self.LINE_WIDTH - len(sender) - 4)
        style = self._get_style(message.role)
        self.console.print(line, style=style)

    def print_message_content(self, content: str | Markdown, end: str = "") -> None:
        """Print message content."""
        # Don't append horizontal lines to content
        self.console.print(content, end=end, soft_wrap=True)

    def print_message_end(self, metadata: MessageMetadata | None = None) -> None:
        """Print message footer with stats."""
        self.console.print()  # Space before stats
        if metadata and (
            metadata.model or metadata.token_usage or metadata.cost is not None
        ):
            parts = []
            if metadata.model:
                parts.append(f"Model: {metadata.model}")
            if metadata.token_usage:
                parts.append(f"Tokens: {metadata.token_usage['total']:,}")
                # Add cost even if metadata.cost is 0.0
                cost = metadata.cost or 0.0
                parts.append(f"Cost: ${cost:.4f}")
            if metadata.response_time:
                parts.append(f"Time: {metadata.response_time:.2f}s")
            stats_line = " • ".join(parts)
            self.console.print(stats_line, style=self.STATS_STYLE)

        self.console.print("─" * self.LINE_WIDTH)

    def _get_sender_name(self, message: ChatMessage) -> str:
        """Get display name for message sender."""
        match message.role:
            case "user":
                return "You"
            case "assistant":
                return message.metadata.name or "Assistant"
            case "system":
                return "System"
            case _:
                return message.role.title()

    def _get_style(self, role: str) -> Style:
        """Get style for message role."""
        match role:
            case "user":
                return self.USER_STYLE
            case "assistant":
                return self.ASSISTANT_STYLE
            case "system":
                return self.SYSTEM_STYLE
            case _:
                return Style()
