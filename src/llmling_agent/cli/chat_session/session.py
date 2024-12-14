"""Interactive chat session implementation."""

from __future__ import annotations

import logging
import traceback
from typing import TYPE_CHECKING

import httpx
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.markdown import Markdown

from llmling_agent.chat_session import ChatSessionManager
from llmling_agent.chat_session.events import (
    SessionEvent,
    SessionEventHandler,
    SessionEventType,
)
from llmling_agent.cli.chat_session import utils
from llmling_agent.cli.chat_session.config import HISTORY_DIR, SessionState
from llmling_agent.cli.chat_session.status import StatusBar
from llmling_agent.commands.base import Command, CommandContext
from llmling_agent.commands.log import SessionLogHandler
from llmling_agent.commands.output import DefaultOutputWriter


if TYPE_CHECKING:
    from llmling_agent import LLMlingAgent
    from llmling_agent.chat_session.base import AgentChatSession


logger = logging.getLogger(__name__)


class CLIEventHandler(SessionEventHandler):
    """Handles session events for CLI interface."""

    async def handle_session_event(self, event: SessionEvent) -> None:
        match event.type:
            case SessionEventType.HISTORY_CLEARED:
                print("\nChat history cleared")
            case SessionEventType.SESSION_RESET:
                print("\nSession reset. Tools restored to default state.")


class InteractiveSession:
    """Interactive chat session using prompt_toolkit."""

    def __init__(
        self,
        agent: LLMlingAgent[str],
        *,
        log_level: int = logging.INFO,
    ) -> None:
        """Initialize interactive session."""
        self.agent = agent
        self._log_level = log_level
        self.console = Console()
        self._output_writer = DefaultOutputWriter()
        # Internal state
        self._session_manager = ChatSessionManager()
        self._chat_session: AgentChatSession | None = None
        self._state = SessionState()
        self.status_bar = StatusBar(self.console)

        # Setup logging
        self._log_handler = SessionLogHandler(self._output_writer)
        self._log_handler.setLevel(log_level)
        logging.getLogger("llmling_agent").addHandler(self._log_handler)
        logging.getLogger("llmling").addHandler(self._log_handler)

        # Setup components
        self._setup_history()
        self._setup_prompt()

    def _setup_history(self) -> None:
        """Setup command history."""
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        history_file = HISTORY_DIR / f"{self.agent.name}.history"
        self._history = FileHistory(str(history_file))

    def _setup_prompt(self) -> None:
        """Setup prompt toolkit session."""
        auto = AutoSuggestFromHistory()
        self._prompt = PromptSession[str](
            "You: ", history=self._history, auto_suggest=auto
        )
        # auto = AutoSuggestFromHistory()
        # completer = CommandCompleter(self._chat_session)
        # self._prompt = PromptSession[str](
        #     "You: ",
        #     history=self._history,
        #     auto_suggest=auto,
        #     completer=completer,
        # )

    def _register_cli_commands(self) -> None:
        """Register CLI-specific commands."""

        async def exit_command(
            ctx: CommandContext,
            args: list[str],
            kwargs: dict[str, str],
        ) -> None:
            """Exit the chat session."""
            raise EOFError

        exit_cmd = Command(
            name="exit",
            description="Exit chat session",
            execute_func=exit_command,
            category="cli",
        )

        assert self._chat_session is not None
        self._chat_session.register_command(exit_cmd)

    async def _handle_input(self, content: str) -> None:
        """Handle user input."""
        if not content.strip():
            return

        try:
            assert self._chat_session is not None

            if content.startswith("/"):
                # Handle command
                result = await self._chat_session.send_message(
                    content,
                    output=DefaultOutputWriter(),
                )
                if result.content:
                    self.console.print(result.content)
                self.status_bar.render(self._state)
                return

            # Handle normal message with streaming
            self.console.print("\nAssistant:", style="bold blue")
            async for chunk in await self._chat_session.send_message(
                content,
                stream=True,
                output=DefaultOutputWriter(),
            ):
                if chunk.content:
                    self.console.print(Markdown(chunk.content))
                # Update tokens for assistant message
                self._state.update_tokens(chunk)

            # Update message count after complete response
            self._state.message_count += 2
            self.status_bar.render(self._state)

        except (httpx.ReadError, GeneratorExit):
            self.console.print("\nConnection interrupted.")
            self.status_bar.render(self._state)
        except EOFError:
            raise
        except Exception as e:  # noqa: BLE001
            error_msg = utils.format_error(e)
            self.console.print(f"\n[red bold]Error:[/] {error_msg}")
            md = Markdown(f"```python\n{traceback.format_exc()}\n```")
            self.console.print("\n[dim]Debug traceback:[/]", md)
            self.status_bar.render(self._state)

    async def start(self) -> None:
        """Start interactive session."""
        try:
            self._chat_session = await self._session_manager.create_session(self.agent)
            self._state.current_model = self._chat_session._model

            # Register event handler AFTER session creation
            cli_handler = CLIEventHandler()
            self._chat_session.add_event_handler(cli_handler)

            self._register_cli_commands()  # Register after session creation
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
                    error_msg = utils.format_error(e)
                    self.console.print(f"\n[red bold]Error:[/] {error_msg}")
                    md = Markdown(f"```python\n{traceback.format_exc()}\n```")
                    self.console.print("\n[dim]Debug traceback:[/]", md)
                    continue

        except Exception as e:  # noqa: BLE001
            self.console.print(f"\n[red bold]Fatal Error:[/] {utils.format_error(e)}")
            md = Markdown(f"```python\n{traceback.format_exc()}\n```")
            self.console.print("\n[dim]Debug traceback:[/]", md)
        finally:
            await self._cleanup()
            await self._show_summary()

    async def _show_welcome(self) -> None:
        """Show welcome message."""
        self.console.print(f"\nStarted chat with {self.agent.name}")
        self.console.print("Type /help for commands or /exit to quit\n")

        # Show initial state
        assert self._chat_session is not None
        tools = self._chat_session.get_tool_states()
        enabled = sum(1 for enabled in tools.values() if enabled)
        text = f"Available tools: {len(tools)} ({enabled} enabled)"
        self.console.print(text, style="dim")

        # Show initial status
        self.status_bar.render(self._state)

    async def _cleanup(self) -> None:
        """Clean up resources."""
        # Remove log handler
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


# Helper function for CLI
async def start_interactive_session(
    agent: LLMlingAgent[str],
    *,
    log_level: int = logging.INFO,
) -> None:
    """Start an interactive chat session."""
    session = InteractiveSession(agent, log_level=log_level)
    await session.start()
