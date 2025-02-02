from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Any, ClassVar

from platformdirs import user_data_dir
from slashed import CommandStore, ExitCommandError, OutputWriter
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Footer, Header

from llmling_agent.messaging.messages import ChatMessage
from llmling_agent_cli.chat_session.models import SessionState
from llmling_agent_commands import get_commands
from llmling_textual.widgets.agent_is_typing import ResponseStatus
from llmling_textual.widgets.chat_view import ChatView
from llmling_textual.widgets.prompt_input import PromptInput


if TYPE_CHECKING:
    from textual.app import ComposeResult

    from llmling_agent import AnyAgent


class TextualOutputWriter(OutputWriter):
    """Output writer that writes to Textual widgets."""

    def __init__(self, chat_view: ChatView):
        self.chat_view = chat_view

    async def print(self, text: str) -> None:
        """Write text to chat view."""
        msg = ChatMessage(content=text, role="system", name="System")
        await self.chat_view.append_chat_message(msg)

    async def print_error(self, text: str) -> None:
        """Write error to chat view."""
        msg = ChatMessage(content=f"Error: {text}", role="system", name="System")
        await self.chat_view.append_chat_message(msg)


class ChatScreen(ModalScreen[None]):
    """Modal screen for chatting with a specific agent."""

    BINDINGS: ClassVar = [
        Binding("escape", "app.pop_screen", "Close", show=True),
        Binding("ctrl+j,ctrl+enter", "send_message", "Send", show=True),
    ]

    DEFAULT_CSS = """
    ChatScreen {
        align: center middle;
    }

    #chat-container {
        width: 90%;
        height: 90%;
        background: $surface;
        border: thick $background;
    }

    #chat-view {
        height: 1fr;
        border: heavy $background;
        margin: 1;
    }

    ResponseStatus {
        height: auto;
        margin: 1;
        background: $surface-darken-2;
    }

    #prompt-input {
        height: 20%;
        dock: bottom;
        border: heavy $background;
        margin: 1;
    }
    """

    def __init__(self, agent: AnyAgent[Any, Any]) -> None:
        super().__init__()
        self.agent = agent
        self._typing_status = ResponseStatus()
        self._typing_status.display = False

        # Initialize command system
        history_dir = pathlib.Path(user_data_dir("llmling", "llmling")) / "cli_history"
        file_path = history_dir / f"{agent.name}.history"
        self.commands = CommandStore(history_file=file_path, enable_system_commands=True)
        self.commands._initialize_sync()
        for cmd in get_commands():
            self.commands.register_command(cmd)

        # Initialize session state
        self._state = SessionState(current_model=self.agent.model_name)

    def compose(self) -> ComposeResult:
        """Create screen layout."""
        yield Header(name=f"Chat with {self.agent.name}")

        with Vertical(id="chat-container"):
            yield ChatView(id="chat-view")
            yield self._typing_status
            yield PromptInput(id="prompt-input")

        yield Footer()

    async def on_mount(self) -> None:
        """Load conversation history when mounted."""
        chat_view = self.query_one(ChatView)
        # Load existing messages
        for msg in self.agent.conversation.chat_messages:
            await chat_view.append_chat_message(msg)
        # Focus input
        self.query_one(PromptInput).focus()

    async def handle_command(self, command_str: str) -> ChatMessage[str]:
        """Handle a slash command."""
        chat_view = self.query_one(ChatView)
        writer = TextualOutputWriter(chat_view)

        try:
            # Create context with our agent's context
            ctx = self.commands.create_context(self.agent.context, output_writer=writer)
            # Execute command
            await self.commands.execute_command(command_str, ctx)
            return ChatMessage(content="", role="system")

        except ExitCommandError:
            # Handle exit command specially
            self.app.pop_screen()
            return ChatMessage(content="Session ended", role="system")

        except Exception as e:  # noqa: BLE001
            return ChatMessage(
                content=f"Command error: {e}", role="system", name="System"
            )

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

        try:
            # Handle commands
            if message.startswith("/"):
                result = await self.handle_command(message[1:])
                if result.content:
                    await chat_view.append_chat_message(result)
                return

            # Show user message
            user_msg = ChatMessage(content=message, role="user", name="You")
            await chat_view.append_chat_message(user_msg)

            # Show typing indicator
            self._typing_status.display = True
            self._typing_status.set_agent_responding()

            # Get response
            response = await self.agent.run(message)
            await chat_view.append_chat_message(response)

            # Update session state
            self._state.message_count += 2
            self._state.update_tokens(response)

        except Exception as e:  # noqa: BLE001
            # Show error in chat
            error_msg = ChatMessage(
                content=f"Error: {e}",
                role="assistant",
                name=self.agent.name,
            )
            await chat_view.append_chat_message(error_msg)

        finally:
            # Re-enable input and hide typing status
            self._typing_status.display = False
            input_widget.submit_ready = True
            input_widget.focus()
