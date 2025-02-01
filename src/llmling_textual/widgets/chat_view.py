from __future__ import annotations

import logfire
from rich.text import Text
from textual.containers import ScrollableContainer
from textual.widgets import Static

from llmling_agent.log import get_logger
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

    MessageWidget > .model {
        color: $text-muted;
        text-style: italic;
        padding-bottom: 1;
    }

    MessageWidget > .content {
        margin-top: 1;
    }
    """

    def __init__(self, message: ChatMessage):
        super().__init__()
        self.message = message
        self.add_class(message.role)
        self.update_content(str(message.content))

    def update_content(self, new_content: str) -> None:
        """Update message content."""
        text = Text()
        text.append(f"{self.message.name}: ", style="bold")
        text.append(new_content)
        self.update(text)


class ChatView(ScrollableContainer):
    """Main chat display widget."""

    DEFAULT_CSS = """
    ChatView {
        width: 100%;
        height: auto;
        padding: 1;
    }
    """

    def __init__(
        self,
        *args,
        name: str | None = None,
        widget_id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
    ):
        super().__init__(
            *args,
            name=name,
            id=widget_id,
            classes=classes,
            disabled=disabled,
        )
        self._current_message: MessageWidget | None = None

    @logfire.instrument("Adding message {message.content}")
    async def append_chat_message(self, message: ChatMessage) -> None:
        """Add a new message to the chat."""
        widget = MessageWidget(message)
        await self.mount(widget)
        widget.scroll_visible()
        self._current_message = widget
        self.refresh(layout=True)

    def start_streaming(self) -> None:
        """Prepare for streaming response."""
        self._current_message = None

    async def update_stream(self, content: str) -> None:
        """Update content of current streaming message."""
        if not self._current_message:
            # Create initial message widget if none exists
            msg = ChatMessage(content=content, role="assistant", name="Assistant")
            self._current_message = MessageWidget(msg)
            await self.mount(self._current_message)

        self._current_message.update_content(content)
        self._current_message.scroll_visible()
        self.refresh(layout=True)
