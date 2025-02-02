from __future__ import annotations

from typing import TYPE_CHECKING

from rich.text import Text
from textual.containers import ScrollableContainer
from textual.widgets import Static

from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from llmling_agent.messaging.messages import ChatMessage


logger = get_logger(__name__)


class MessageWidget(Static):
    """Individual message in the chat."""

    DEFAULT_CSS = """
    MessageWidget {
        height: auto;
        margin: 0 1;
        padding: 1;
        border-title-align: left;
        border-title-color: $text-muted;
        border-title-background: $surface;
        border: heavy $primary;
    }

    MessageWidget.user {
        border: heavy $primary;
    }

    MessageWidget.assistant {
        border: heavy $success;
    }

    MessageWidget.system {
        border: heavy $warning;
    }
    """

    def __init__(self, message: ChatMessage):
        super().__init__("")
        self.chat_message = message
        self.add_class(message.role)
        text = Text()
        text.append(f"{message.name}: ", style="bold")
        text.append(str(message.content))
        self.update(text)


class ChatView(ScrollableContainer):
    """Chat message display."""

    DEFAULT_CSS = """
    ChatView {
        width: 100%;
        height: auto;
        padding: 1;
    }
    """

    async def append_chat_message(self, message: ChatMessage) -> None:
        """Add a new message to the chat."""
        widget = MessageWidget(message)
        await self.mount(widget)
        widget.scroll_visible()
        self.refresh(layout=True)
