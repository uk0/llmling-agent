"""Message flow widget."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.text import Text
from textual.widgets import Static


if TYPE_CHECKING:
    from rich.console import RenderableType

    from llmling_agent.talk.talk import Talk


class MessageFlowWidget(Static):
    """Display a single message flow with source, targets, and content."""

    DEFAULT_CSS = """
    MessageFlowWidget {
        padding: 1;
        margin: 1;
        border: round $primary;
        height: auto;
    }

    MessageFlowWidget.queued {
        border: round $warning;
    }

    MessageFlowWidget > .header {
        color: $text-muted;
        padding-bottom: 1;
    }

    MessageFlowWidget > .content {
        padding-left: 1;
    }
    """

    def __init__(
        self,
        event: Talk.ConnectionProcessed,
        id: str | None = None,  # noqa: A002
        classes: str | None = None,
    ):
        super().__init__(id=id, classes=classes)
        self.event = event
        if event.queued:
            self.add_class("queued")

    def render(self) -> RenderableType:
        """Render the message flow."""
        # Create header with timestamp and routing info
        header = Text()
        header.append(f"[{self.event.timestamp:%H:%M:%S}] ", style="dim")
        header.append(self.event.source.name, style="bold")
        header.append(" → ")
        header.append(", ".join(t.name for t in self.event.targets), style="italic")
        if self.event.queued:
            header.append(" [QUEUED]", style="yellow")

        # Message content
        content = Text(str(self.event.message.content))

        return Text.assemble(header, "\n", content, end="")


if __name__ == "__main__":
    from textualicious import show

    from llmling_agent import Agent, ChatMessage
    from llmling_agent.talk.talk import Talk

    agent = Agent[None]()
    message = ChatMessage("Hello, world!", "user")
    event = Talk.ConnectionProcessed(message, agent, [], False, "run")
    show(MessageFlowWidget(event))
