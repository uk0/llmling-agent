from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Footer, Header, TextArea

from llmling_agent.models.messages import ChatMessage
from llmling_textual.widgets.chat_view import ChatView


class ChatScreen(ModalScreen[None]):
    """Modal screen for chatting with a specific agent."""

    BINDINGS: ClassVar = [
        Binding("escape", "pop_screen", "Close chat", show=True),
        Binding("ctrl+enter", "send_message", "Send", show=True),
    ]

    def __init__(self, agent) -> None:
        super().__init__()
        self.agent = agent

    def compose(self) -> ComposeResult:
        yield Header(name=f"Chat with {self.agent.name}")
        with Vertical():
            yield ChatView(id="agent-chat")
            yield TextArea(id="chat-input")
        yield Footer()

    def action_send_message(self) -> None:
        """Send message to agent."""
        if input_widget := self.query_one(TextArea):
            message = input_widget.text.strip()
            if message:
                input_widget.clear()
                # Create task for agent interaction
                self.run_worker(self.run_agent(message))

    async def run_agent(self, message: str) -> None:
        """Run agent with message and show response."""
        chat_view = self.query_one(ChatView)

        # Show user message
        user_msg = ChatMessage(content=message, role="user", name="You")
        await chat_view.append_chat_message(user_msg)

        try:
            # Get agent response
            response = await self.agent.run(message)
            await chat_view.append_chat_message(response)
        except Exception as e:  # noqa: BLE001
            # Show error in chat
            error_msg = ChatMessage(
                content=f"Error: {e}", role="assistant", name=self.agent.name
            )
            await chat_view.append_chat_message(error_msg)
