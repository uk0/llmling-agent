from __future__ import annotations

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
    """

    def __init__(self, message: ChatMessage):
        super().__init__("")
        self.message = message
        self.add_class(message.role)
        self._content = ""
        self.update_content(str(message.content))

    def update_content(self, new_content: str) -> None:
        """Update message content."""
        self._content = new_content
        text = Text()
        text.append(f"{self.message.name}: ", style="bold")
        text.append(self._content)
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

    async def append_chat_message(self, message: ChatMessage) -> None:
        """Add a new message to the chat."""
        widget = MessageWidget(message)
        await self.mount(widget)
        widget.scroll_visible()
        self.refresh(layout=True)

    def start_streaming(self) -> None:
        """Start a new streaming message."""
        # Create initial assistant message
        msg = ChatMessage(content="", role="assistant", name="Assistant")
        self._current_message = MessageWidget(msg)
        self.mount(self._current_message)
        self._current_message.scroll_visible()

    async def update_stream(self, chunk: str) -> None:
        """Update current streaming message with new chunk."""
        if not self._current_message:
            self.start_streaming()
        assert self._current_message is not None
        self._current_message.update_content(chunk)
        self._current_message.scroll_visible()

    def finalize_stream(self, final_message: ChatMessage) -> None:
        """Update streaming message with final metadata."""
        if self._current_message:
            self._current_message.message = final_message
            self._current_message.update_content(str(final_message.content))
            self._current_message = None
