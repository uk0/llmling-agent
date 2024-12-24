"""Interactive chat session implementation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import httpx
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from rich.console import Console
from rich.markdown import Markdown
from slashed import ExitCommandError
from slashed.log import SessionLogHandler

from llmling_agent.chat_session import ChatSessionManager
from llmling_agent.chat_session.output import DefaultOutputWriter
from llmling_agent.chat_session.welcome import create_welcome_messages
from llmling_agent.models.messages import ChatMessage
from llmling_agent_cli.chat_session.completion import PromptToolkitCompleter
from llmling_agent_cli.chat_session.formatting import MessageFormatter
from llmling_agent_cli.chat_session.history import SessionHistory


if TYPE_CHECKING:
    from llmling_agent import LLMlingAgent
    from llmling_agent.chat_session.base import AgentChatSession
    from llmling_agent.chat_session.events import HistoryClearedEvent, SessionResetEvent
    from llmling_agent.models.agents import ToolCallInfo
    from llmling_agent.tools.base import ToolInfo


logger = logging.getLogger(__name__)


class InteractiveSession:
    """Interactive chat session using prompt_toolkit."""

    def __init__(
        self,
        agent: LLMlingAgent[Any, str],
        *,
        log_level: int = logging.WARNING,
        show_log_in_chat: bool = False,
        stream: bool = False,
        render_markdown: bool = False,
    ):
        """Initialize interactive session."""
        self.agent = agent
        self._stream = stream
        self._render_markdown = render_markdown

        # Setup console and formatting
        self.console = Console()
        self.formatter = MessageFormatter(self.console)

        # Setup session management
        self._session_manager = ChatSessionManager()
        self._chat_session: AgentChatSession | None = None
        self._prompt: PromptSession | None = None

        # Setup logging
        self._log_handler = None
        if show_log_in_chat:
            self._log_handler = SessionLogHandler(DefaultOutputWriter())
            self._log_handler.setLevel(log_level)
            logging.getLogger("llmling_agent").addHandler(self._log_handler)
            logging.getLogger("llmling").addHandler(self._log_handler)

    def _connect_signals(self):
        """Connect to chat session signals."""
        assert self._chat_session is not None
        self._chat_session.history_cleared.connect(self._on_history_cleared)
        self._chat_session.session_reset.connect(self._on_session_reset)
        self._chat_session.tool_added.connect(self._on_tool_added)
        self._chat_session.tool_removed.connect(self._on_tool_removed)
        self._chat_session.tool_changed.connect(self._on_tool_changed)
        self._chat_session._agent.tool_used.connect(self._on_tool_call)

    def _on_tool_added(self, tool: ToolInfo):
        """Handle tool addition."""
        self.console.print(f"\nTool added: {tool.name}")

    def _on_tool_removed(self, tool_name: str):
        """Handle tool removal."""
        self.console.print(f"\nTool removed: {tool_name}")

    def _on_tool_call(self, tool_call: ToolCallInfo):
        """Handle tool usage signal."""
        logger.debug("Tool call received: %s", tool_call.tool_name)
        self.formatter.print_tool_call(tool_call)

    def _on_tool_changed(self, name: str, tool: ToolInfo):
        """Handle tool state changes."""
        state = "enabled" if tool.enabled else "disabled"
        self.console.print(f"\nTool '{name}' {state}")

    def _on_history_cleared(self, event: HistoryClearedEvent):
        """Handle history cleared event."""
        self.console.print("\nChat history cleared")

    def _on_session_reset(self, event: SessionResetEvent):
        """Handle session reset event."""
        self.console.print("\nSession reset. Tools restored to default state.")

    def _setup_prompt(self):
        """Setup prompt toolkit session."""
        assert self._chat_session is not None

        history = SessionHistory(self._chat_session)
        completer = PromptToolkitCompleter(self._chat_session._command_store._commands)

        self._prompt = PromptSession[str](
            "You: ",
            history=history,
            auto_suggest=AutoSuggestFromHistory(),
            completer=completer,
        )

    async def _handle_input(self, content: str):
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
                        self.formatter.print_message_start(result)
                        self.formatter.print_message_content(result.content)
                        self.formatter.print_message_end(result.metadata)
                except ExitCommandError:
                    self.formatter.print_exit()
                    raise EOFError  # noqa: B904
                return

            # Show user message
            user_msg: ChatMessage[str] = ChatMessage(content=content, role="user")
            self.formatter.print_message_start(user_msg)
            self.formatter.print_message_content(content)
            self.formatter.print_message_end(None)

            if self._stream:
                buffer = ""
                first_chunk = True
                async for chunk in await self._chat_session.send_message(
                    content,
                    stream=True,
                    output=writer,
                ):
                    if chunk.content:
                        if first_chunk:
                            self.formatter.print_message_start(chunk)
                            first_chunk = False

                        new_content = chunk.content
                        if new_content != buffer:
                            diff = new_content[len(buffer) :]
                            self.formatter.print_message_content(diff, end="")
                            buffer = new_content

                self.console.print()  # New line after streaming
                self.formatter.print_message_end(chunk.metadata)
            else:
                # Non-streaming mode
                result = await self._chat_session.send_message(content, output=writer)
                if result.content:
                    self.formatter.print_message_start(result)
                    content_to_print = (
                        Markdown(result.content)
                        if self._render_markdown
                        else result.content
                    )
                    self.formatter.print_message_content(content_to_print)
                    self.formatter.print_message_end(result.metadata)

        except (httpx.ReadError, GeneratorExit):
            self.formatter.print_connection_error()
        except EOFError:
            raise
        except Exception as e:  # noqa: BLE001
            self.formatter.print_error(e, show_traceback=True)

    async def start(self):
        """Start interactive session."""
        try:
            self._chat_session = await self._session_manager.create_session(self.agent)
            self._connect_signals()
            self._setup_prompt()

            # Show welcome message
            welcome_info = create_welcome_messages(
                self._chat_session, streaming=self._stream, rich_format=True
            )
            self.formatter.print_welcome(welcome_info)

            # Main interaction loop
            while True:
                try:
                    assert self._prompt
                    if user_input := await self._prompt.prompt_async():
                        await self._handle_input(user_input)
                except KeyboardInterrupt:
                    self.console.print("\nUse /exit to quit")
                    continue
                except EOFError:
                    break
                except Exception as e:  # noqa: BLE001
                    self.formatter.print_error(e, show_traceback=True)
                    continue

        except Exception as e:  # noqa: BLE001
            self.formatter.print_error(e, show_traceback=True)
        finally:
            await self._cleanup()

    async def _cleanup(self):
        """Clean up resources."""
        if self._log_handler:
            logging.getLogger("llmling_agent").removeHandler(self._log_handler)
            logging.getLogger("llmling").removeHandler(self._log_handler)


# Helper function for CLI
async def start_interactive_session(
    agent: LLMlingAgent[str, str],
    *,
    log_level: int = logging.WARNING,
    stream: bool = False,
):
    """Start an interactive chat session."""
    session = InteractiveSession(
        agent,
        log_level=log_level,
        stream=stream,
    )
    await session.start()
