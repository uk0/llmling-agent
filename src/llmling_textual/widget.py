from __future__ import annotations

from typing import TYPE_CHECKING

import logfire
from slashed.log import get_logger
from textual.containers import ScrollableContainer
from textual.widgets import Static


if TYPE_CHECKING:
    from textual.app import ComposeResult

    from llmling_agent.models.messages import ChatMessage


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
        self.border_title = self.message.name or self.message.role.title()

    def compose(self) -> ComposeResult:
        """Create message layout."""
        if self.message.model:
            yield Static(f"using {self.message.model}", classes="model")
        # Initialize with empty content for assistant, actual content for others
        initial_content = "" if self.message.role == "assistant" else self.message.content
        yield Static(initial_content, id="message_content", classes="content")

    @logfire.instrument("Updating content to {new_content}")
    def update_content(self, new_content: str):
        """Update message content."""
        if content_widget := self.query_one("#message_content", Static):
            content_widget.update(new_content)
            logger.debug("Content widget updated successfully")
        else:
            logger.warning("No content widget found!")


class ChatView(ScrollableContainer):
    """Main chat display widget."""

    DEFAULT_CSS = """
    ChatView {
        height: 1fr;
        width: 100%;
        background: $surface;
        padding: 1;
    }
    """

    def __init__(self):
        super().__init__()
        self._current_message: MessageWidget | None = None

    @logfire.instrument("Adding message {message.content}")
    async def add_message(self, message: ChatMessage) -> MessageWidget:
        """Add a new message to the chat."""
        widget = MessageWidget(message)
        await self.mount(widget)
        widget.scroll_visible()
        self._current_message = widget
        return widget

    def update_stream(self, content: str):
        """Update content of current streaming message."""
        if self._current_message:
            logger.debug("Updating stream: %r", content)
            self._current_message.update_content(content)
            self._current_message.scroll_visible()
