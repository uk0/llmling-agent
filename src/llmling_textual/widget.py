"""Chat widgets for LLMling agent."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.text import Text
from textual.containers import ScrollableContainer
from textual.reactive import reactive
from textual.widgets import Static


if TYPE_CHECKING:
    from rich.console import RenderableType
    from textual.app import ComposeResult

    from llmling_agent.models.messages import ChatMessage


class MessageWidget(Static):
    """Individual message in the chat."""

    content = reactive("")

    DEFAULT_CSS = """
    MessageWidget {
        height: auto;
        margin: 0 1;
        padding: 1;
        border-title-align: left;
        border-title-color: $text-muted;
        border-title-background: $surface;
        box-sizing: border-box;
        border: ascii $primary;   # This gives continuous lines
    }

    MessageWidget.user {
        border: ascii $primary;
    }

    MessageWidget.assistant {
        border: ascii $success;
    }

    MessageWidget.system {
        border: ascii $warning;
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
        super().__init__()  # Call this first!
        self.message = message
        self.add_class(message.role)
        self.border_title = self.message.name or self.message.role.title()

        # Create container for model info and content
        self.model_info = (
            Static(f"using {self.message.model}", classes="model")
            if self.message.model
            else None
        )
        self.content_widget = Static(id="message_content", classes="content")
        self.content = message.content  # This will trigger watch_content

    def compose(self) -> ComposeResult:
        """Create message layout."""
        if self.model_info:
            yield self.model_info
        yield self.content_widget

    def watch_content(self, new_content: str) -> None:
        """React to content changes."""
        self.content_widget.update(new_content)

    def render(self) -> RenderableType:
        """Not used anymore as we're using child widgets."""
        return Text("")


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

    async def add_message(self, message: ChatMessage) -> MessageWidget:
        """Add a new message to the chat."""
        widget = MessageWidget(message)
        await self.mount(widget)
        widget.scroll_visible()
        self._current_message = widget
        return widget

    async def update_stream(self, content: str):
        """Update content of current streaming message."""
        if self._current_message:
            self._current_message.content = content
            self._current_message.scroll_visible()
