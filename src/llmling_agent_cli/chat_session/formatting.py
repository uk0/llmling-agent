from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from rich.console import Console
from rich.style import Style
from rich.traceback import Traceback

from llmling_agent.chat_session.exceptions import format_error


if TYPE_CHECKING:
    from rich.markdown import Markdown

    from llmling_agent.chat_session.welcome import WelcomeInfo
    from llmling_agent.models.agents import ToolCallInfo
    from llmling_agent.models.messages import ChatMessage, MessageMetadata


class MessageFormatter:
    """Format chat messages and related content for CLI display."""

    LINE_WIDTH: ClassVar[int] = 80
    USER_STYLE: ClassVar[Style] = Style(color="blue")
    ASSISTANT_STYLE: ClassVar[Style] = Style(color="green")
    SYSTEM_STYLE: ClassVar[Style] = Style(color="yellow")
    STATS_STYLE: ClassVar[Style] = Style(dim=True)
    ERROR_STYLE: ClassVar[Style] = Style(color="red", bold=True)
    TOOL_STYLE: ClassVar[Style] = Style(color="yellow")

    def __init__(self, console: Console | None = None):
        self.console = console or Console()

    def print_message_start(self, message: ChatMessage):
        """Print message header."""
        self.console.print()
        sender = self._get_sender_name(message)
        line = f"─── {sender} " + "─" * (self.LINE_WIDTH - len(sender) - 4)
        style = self._get_style(message.role)
        self.console.print(line, style=style)

    def print_message_content(self, content: str | Markdown, end: str = ""):
        """Print message content."""
        self.console.print(content, end=end, soft_wrap=True)

    def print_message_end(self, metadata: MessageMetadata | None = None):
        """Print message footer with stats."""
        self.console.print()
        if metadata and (
            metadata.model or metadata.token_usage or metadata.cost is not None
        ):
            parts = []
            if metadata.model:
                parts.append(f"Model: {metadata.model}")
            if metadata.token_usage:
                parts.append(f"Tokens: {metadata.token_usage['total']:,}")
                cost = metadata.cost or 0.0
                parts.append(f"Cost: ${cost:.4f}")
            if metadata.response_time:
                parts.append(f"Time: {metadata.response_time:.2f}s")

            stats_line = " • ".join(parts)
            self.console.print(stats_line, style=self.STATS_STYLE)

        self.console.print("─" * self.LINE_WIDTH)

    def print_error(self, error: Exception, show_traceback: bool = False):
        """Print error message with optional traceback."""
        error_msg = format_error(error)
        self.console.print(f"\n[red bold]Error:[/] {error_msg}")
        if show_traceback:
            self.console.print("\n[dim]Debug traceback:[/]")
            self.console.print(
                Traceback.from_exception(
                    type(error),
                    error,
                    error.__traceback__,
                    show_locals=True,
                    width=self.LINE_WIDTH,
                )
            )

    def print_tool_call(self, tool_call: ToolCallInfo):
        """Print tool call information."""
        self.console.print()
        self.console.print("Tool Call:", style=self.TOOL_STYLE)
        self.console.print(f"  Name: {tool_call.tool_name}")
        self.console.print(f"  Args: {tool_call.args}")
        self.console.print(f"  Result: {tool_call.result}")

    def print_welcome(self, welcome_info: WelcomeInfo):
        """Print welcome message sections.

        Args:
            welcome_info: Welcome information from create_welcome_messages()
        """
        for title, lines in welcome_info.all_sections():
            if title:  # Skip empty section titles
                self.console.print(f"\n[bold]{title}[/]")
            for line in lines:
                self.console.print(line)

    def print_session_summary(
        self, messages: int, tokens: dict[str, int], cost: float, duration: str
    ):
        """Print end of session summary."""
        self.console.print("\nSession Summary:")
        self.console.print(f"Messages: {messages}")
        token_info = (
            f"Total tokens: {tokens['total']:,} "
            f"(Prompt: {tokens['prompt']:,}, "
            f"Completion: {tokens['completion']:,})"
        )
        self.console.print(token_info)
        self.console.print(f"Total cost: ${cost:.6f}")
        self.console.print(f"Duration: {duration}")

    def print_connection_error(self):
        """Print connection error message."""
        self.console.print("\nConnection interrupted.", style=self.ERROR_STYLE)

    def print_exit(self):
        """Print exit message."""
        self.console.print("\nGoodbye!")

    def print_chain_status(self, agent_names: list[str], status: str):
        """Print chain processing status."""
        self.console.print(f"\n[dim]Chain ({', '.join(agent_names)}): {status}[/]")

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
