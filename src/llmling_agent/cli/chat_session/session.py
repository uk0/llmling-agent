"""Interactive chat session implementation."""

from __future__ import annotations

import logging
import traceback
from typing import TYPE_CHECKING

import httpx
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from rich.console import Console
from rich.markdown import Markdown

from llmling_agent.chat_session import ChatSessionManager
from llmling_agent.chat_session.exceptions import format_error
from llmling_agent.chat_session.models import SessionState
from llmling_agent.chat_session.output import DefaultOutputWriter
from llmling_agent.chat_session.welcome import create_welcome_messages
from llmling_agent.cli.chat_session.completion import PromptToolkitCompleter
from llmling_agent.cli.chat_session.history import SessionHistory
from llmling_agent.cli.chat_session.status import render_status_bar
from llmling_agent.commands.exceptions import ExitCommandError
from llmling_agent.commands.log import SessionLogHandler
from llmling_agent.ui.status import StatusBar


if TYPE_CHECKING:
    from llmling_agent import LLMlingAgent
    from llmling_agent.chat_session.base import AgentChatSession
    from llmling_agent.chat_session.events import HistoryClearedEvent, SessionResetEvent
    from llmling_agent.tools.base import ToolInfo


logger = logging.getLogger(__name__)


class InteractiveSession:
    """Interactive chat session using prompt_toolkit."""

    def __init__(
        self,
        agent: LLMlingAgent[str],
        *,
        log_level: int = logging.WARNING,
        show_log_in_chat: bool = False,
        stream: bool = False,
    ) -> None:
        """Initialize interactive session."""
        self.agent = agent
        self._log_level = log_level
        self.console = Console()
        self._stream = stream
        self._output_writer = DefaultOutputWriter()
        # Internal state
        self._session_manager = ChatSessionManager()
        self._chat_session: AgentChatSession | None = None
        self._state = SessionState()
        self.status_bar = StatusBar()
        self._prompt: PromptSession | None = None
        # Setup logging
        self._log_handler = None
        if show_log_in_chat:
            self._log_handler = SessionLogHandler(self._output_writer)
            self._log_handler.setLevel(log_level)
            logging.getLogger("llmling_agent").addHandler(self._log_handler)
            logging.getLogger("llmling").addHandler(self._log_handler)

    def _connect_signals(self) -> None:
        """Connect to chat session signals."""
        assert self._chat_session is not None

        # Connect to signals with bound methods
        self._chat_session.history_cleared.connect(self._on_history_cleared)
        self._chat_session.session_reset.connect(self._on_session_reset)
        self._chat_session.tool_added.connect(self._on_tool_added)
        self._chat_session.tool_removed.connect(self._on_tool_removed)
        self._chat_session.tool_changed.connect(self._on_tool_changed)

    def _on_tool_added(self, tool: ToolInfo) -> None:
        """Handle tool addition."""
        self.console.print(f"\nTool added: {tool.name}")
        self.update_status_bar()

    def _on_tool_removed(self, tool_name: str) -> None:
        """Handle tool removal."""
        self.console.print(f"\nTool removed: {tool_name}")
        self.update_status_bar()

    def _on_tool_changed(self, name: str, tool: ToolInfo) -> None:
        """Handle tool state changes."""
        state = "enabled" if tool.enabled else "disabled"
        self.console.print(f"\nTool '{name}' {state}")
        self.update_status_bar()

    def _on_history_cleared(self, event: HistoryClearedEvent) -> None:
        """Handle history cleared event."""
        self.console.print("\nChat history cleared")

    def _on_session_reset(self, event: SessionResetEvent) -> None:
        """Handle session reset event."""
        self.console.print("\nSession reset. Tools restored to default state.")

    def _setup_prompt(self) -> None:
        """Setup prompt toolkit session."""
        auto = AutoSuggestFromHistory()
        history = SessionHistory(self.session)

        self._prompt = PromptSession[str]("You: ", history=history, auto_suggest=auto)

    async def _handle_input(self, content: str) -> None:
        """Handle user input."""
        if not content.strip():
            return
        writer = DefaultOutputWriter()
        try:
            assert self._chat_session is not None

            if content.startswith("/"):
                try:
                    result = await self._chat_session.send_message(content, output=writer)
                    if result.content:
                        self.console.print(result.content)
                except ExitCommandError as e:
                    # Handle clean exit
                    self.console.print("\nGoodbye!")
                    raise EOFError from e
                return

            # Handle normal message with or without streaming
            self.console.print("\nAssistant:", style="bold blue")

            if self._stream:
                # Existing streaming code
                async for chunk in await self._chat_session.send_message(
                    content,
                    stream=True,
                    output=writer,
                ):
                    if chunk.content:
                        self.console.print(Markdown(chunk.content))
                    self._state.update_tokens(chunk)
            else:
                # Non-streaming mode
                result = await self._chat_session.send_message(content, output=writer)
                if result.content:
                    self.console.print(Markdown(result.content))
                self._state.update_tokens(result)

            # Update message count after complete response
            self._state.message_count += 2
        except (httpx.ReadError, GeneratorExit):
            self.console.print("\nConnection interrupted.")
        except EOFError:
            raise
        except Exception as e:  # noqa: BLE001
            error_msg = format_error(e)
            self.console.print(f"\n[red bold]Error:[/] {error_msg}")
            md = Markdown(f"```python\n{traceback.format_exc()}\n```")
            self.console.print("\n[dim]Debug traceback:[/]", md)
        finally:
            self.update_status_bar()

    def update_status_bar(self):
        """Update and render status bar."""
        self.status_bar.update(self._state)
        render_status_bar(self.status_bar, self.console)

    @property
    def session(self) -> AgentChatSession:
        """Get the current chat session."""
        assert self._chat_session is not None
        return self._chat_session

    async def start(self) -> None:
        """Start interactive session."""
        try:
            self._chat_session = await self._session_manager.create_session(self.agent)
            self._state.current_model = self._chat_session._model
            self._connect_signals()

            completer = PromptToolkitCompleter(
                self._chat_session._command_store._commands
            )
            self._setup_prompt()
            assert self._prompt
            self._prompt.completer = completer  # Update the prompt's completer
            # Register event handler AFTER session creation

            await self._show_welcome()

            while True:
                try:
                    if user_input := await self._prompt.prompt_async():
                        await self._handle_input(user_input)
                except KeyboardInterrupt:
                    self.console.print("\nUse /exit to quit")
                    continue
                except EOFError:
                    break
                except Exception as e:  # noqa: BLE001
                    error_msg = format_error(e)
                    self.console.print(f"\n[red bold]Error:[/] {error_msg}")
                    md = Markdown(f"```python\n{traceback.format_exc()}\n```")
                    self.console.print("\n[dim]Debug traceback:[/]", md)
                    continue

        except Exception as e:  # noqa: BLE001
            self.console.print(f"\n[red bold]Fatal Error:[/] {format_error(e)}")
            md = Markdown(f"```python\n{traceback.format_exc()}\n```")
            self.console.print("\n[dim]Debug traceback:[/]", md)
        finally:
            await self._cleanup()
            await self._show_summary()

    async def _show_welcome(self) -> None:
        """Show welcome message."""
        assert self._chat_session is not None, (
            "Chat session must be initialized before showing welcome"
        )

        welcome_info = create_welcome_messages(self._chat_session, streaming=self._stream)
        for _, lines in welcome_info.all_sections():
            for line in lines:
                self.console.print(line)
        # Show initial status
        self.update_status_bar()

    async def _cleanup(self) -> None:
        """Clean up resources."""
        # Remove log handler
        if self._log_handler:
            logging.getLogger("llmling_agent").removeHandler(self._log_handler)
            logging.getLogger("llmling").removeHandler(self._log_handler)

        if self._chat_session:
            # Any cleanup needed for chat session
            pass

    async def _show_summary(self) -> None:
        """Show session summary."""
        if self._state.message_count > 0:
            self.console.print("\nSession Summary:")
            self.console.print(f"Messages: {self._state.message_count}")
            token_info = (
                f"Total tokens: {self._state.total_tokens:,} "
                f"(Prompt: {self._state.prompt_tokens:,}, "
                f"Completion: {self._state.completion_tokens:,})"
            )
            self.console.print(token_info)
            if self._state.total_cost > 0:
                cost_info = f"Total cost: ${self._state.total_cost:.6f}"
                self.console.print(cost_info)


# Helper function for CLI
async def start_interactive_session(
    agent: LLMlingAgent[str],
    *,
    log_level: int = logging.WARNING,
    stream: bool = False,
) -> None:
    """Start an interactive chat session."""
    session = InteractiveSession(agent, log_level=log_level, stream=stream)
    await session.start()
