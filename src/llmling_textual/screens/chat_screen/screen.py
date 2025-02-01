from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Footer, Header

from llmling_agent.messaging.messages import ChatMessage
from llmling_textual.widgets.agent_is_typing import ResponseStatus
from llmling_textual.widgets.chat_view import ChatView
from llmling_textual.widgets.prompt_input import PromptInput


if TYPE_CHECKING:
    from textual.app import ComposeResult

    from llmling_agent import AnyAgent


class ChatScreen(ModalScreen[None]):
    """Modal screen for chatting with a specific agent."""

    BINDINGS: ClassVar = [
        Binding("escape", "app.pop_screen", "Close", show=True),
        Binding("ctrl+j,ctrl+enter", "send_message", "Send", show=True),
        Binding("ctrl+c", "app.pop_screen", "Cancel", show=False),
    ]

    DEFAULT_CSS = """
    ChatScreen {
        align: center middle;
    }

    #chat-container {
        width: 95%;
        height: 95%;
        background: $surface;
        border: thick $background;
    }

    #chat-scroll {
        height: 80%;
        margin: 1;
    }

    #chat-input {
        height: 20%;
        dock: bottom;
        border: heavy $background;
        background: $surface;
        margin: 1;
    }

    #chat-view {
        height: auto;
        border: heavy $background;
        background: $surface-darken-1;
    }

    ResponseStatus {
        height: auto;
        dock: bottom;
        padding: 1;
        background: $surface-darken-2;
    }
    """

    def __init__(self, agent: AnyAgent[Any, Any]) -> None:
        """Initialize chat screen."""
        super().__init__()
        self.agent = agent
        self._typing_status = ResponseStatus()
        self._typing_status.display = False

    def compose(self) -> ComposeResult:
        """Create screen layout."""
        yield Header(name=f"Chat with {self.agent.name}")

        with Vertical(id="chat-container"):
            with VerticalScroll(id="chat-scroll"):
                yield ChatView(widget_id="chat-view")
                yield self._typing_status
            yield PromptInput(widget_id="chat-input")

        yield Footer()

    async def on_prompt_input_prompt_submitted(
        self, event: PromptInput.PromptSubmitted
    ) -> None:
        """Handle submitted prompt."""
        message = event.text.strip()
        if not message:
            self.notify("Cannot send empty message", severity="error")
            return

        chat_view = self.query_one(ChatView)
        input_widget = event.prompt_input

        # Disable input while processing
        input_widget.submit_ready = False

        # Show user message
        user_msg = ChatMessage(content=message, role="user", name="You")
        await chat_view.append_chat_message(user_msg)

        # Show typing indicator
        self._typing_status.display = True
        self._typing_status.set_agent_responding()

        try:
            # Initialize stream response
            chat_view.start_streaming()

            # Get agent response
            async with self.agent.run_stream(message) as stream:
                current_content = ""
                async for chunk in stream.stream():
                    current_content += chunk
                    await chat_view.update_stream(current_content)

                response = ChatMessage(
                    content=current_content,
                    role="assistant",
                    name=self.agent.name,
                    model=stream.model_name,
                )
                await chat_view.append_chat_message(response)

        except Exception as e:  # noqa: BLE001
            # Show error in chat
            error_msg = ChatMessage(
                content=f"Error: {e}", role="assistant", name=self.agent.name
            )
            await chat_view.append_chat_message(error_msg)

        finally:
            # Re-enable input and hide typing status
            self._typing_status.display = False
            input_widget.submit_ready = True
            input_widget.focus()

    def on_key(self, event) -> None:
        """Handle key events."""
        if event.key == "escape":
            self.app.pop_screen()
        elif event.key == "ctrl+j":
            self.action_send_message()

    def action_send_message(self) -> None:
        """Action to send the current message."""
        if input_widget := self.query_one(PromptInput):
            input_widget.action_submit_prompt()

    async def on_mount(self) -> None:
        """Focus input when screen mounts."""
        self.query_one(PromptInput).focus()
