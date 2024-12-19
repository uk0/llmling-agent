"""Core chat session implementation."""

from __future__ import annotations

from datetime import datetime
import pathlib
from typing import TYPE_CHECKING, Any, Literal, overload
from uuid import UUID, uuid4

from platformdirs import user_data_dir
from psygnal import Signal
from sqlalchemy import desc
from sqlmodel import Session, select

from llmling_agent.chat_session.events import (
    HistoryClearedEvent,
    SessionResetEvent,
)
from llmling_agent.chat_session.exceptions import ChatSessionConfigError
from llmling_agent.chat_session.models import ChatSessionMetadata, SessionState
from llmling_agent.chat_session.output import DefaultOutputWriter, OutputWriter
from llmling_agent.commands import CommandStore
from llmling_agent.commands.base import BaseCommand, CommandContext
from llmling_agent.commands.exceptions import CommandError, ExitCommandError
from llmling_agent.log import get_logger
from llmling_agent.models.messages import ChatMessage, MessageMetadata
from llmling_agent.pydantic_ai_utils import extract_token_usage_and_cost
from llmling_agent.storage import engine
from llmling_agent.storage.models import CommandHistory
from llmling_agent.tools.base import ToolInfo


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from pydantic_ai import messages

    from llmling_agent import LLMlingAgent
    from llmling_agent.tools.manager import ToolManager


logger = get_logger(__name__)
HISTORY_DIR = pathlib.Path(user_data_dir("llmling", "llmling")) / "cli_history"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)


class AgentChatSession:
    """Manages an interactive chat session with an agent.

    This class:
    1. Manages agent configuration (tools, model)
    2. Handles conversation flow
    3. Tracks session state and metadata
    """

    history_cleared = Signal(HistoryClearedEvent)
    session_reset = Signal(SessionResetEvent)
    tool_added = Signal(ToolInfo)
    tool_removed = Signal(str)  # tool_name
    tool_changed = Signal(str, ToolInfo)  # name, new_info

    def __init__(
        self,
        agent: LLMlingAgent[str],
        *,
        session_id: UUID | str | None = None,
        model_override: str | None = None,
    ) -> None:
        """Initialize chat session.

        Args:
            agent: The LLMling agent to use
            session_id: Optional session ID (generated if not provided)
            model_override: Optional model override for this session
        """
        # Basic setup that doesn't need async
        match session_id:
            case str():
                self.id = UUID(session_id)
            case UUID():
                self.id = session_id
            case None:
                self.id = uuid4()
        self._agent = agent
        # forward ToolManager signals to ours

        self._agent.tools.events.added.connect(self.tool_added.emit)
        self._agent.tools.events.removed.connect(self.tool_removed.emit)
        self._agent.tools.events.changed.connect(self.tool_changed.emit)
        self._model = model_override or agent.model_name
        self._history: list[messages.ModelMessage] = []
        self._commands: list[str] = []
        self._history_file = HISTORY_DIR / f"{agent.name}.history"
        self._initialized = False  # Track initialization state

        # Initialize basic structures
        self._command_store = CommandStore()
        self.start_time = datetime.now()
        self._state = SessionState(current_model=self._model)

    def _ensure_initialized(self) -> None:
        """Check if session is initialized."""
        if not self._initialized:
            msg = "Session not initialized. Call initialize() first."
            raise RuntimeError(msg)

    async def initialize(self) -> None:
        """Initialize async resources and load data."""
        if self._initialized:
            return

        # Load command history
        try:
            if self._history_file.exists():
                self._commands = self._history_file.read_text().splitlines()
        except Exception:
            logger.exception("Failed to load command history")
            self._commands = []

        # Initialize tool states
        self._tool_states = self._agent.tools.list_tools()

        # Initialize command system
        self._command_store.register_builtin_commands()

        # Any other async initialization...

        self._initialized = True
        logger.debug(
            "Initialized chat session %s for agent %s", self.id, self._agent.name
        )

    def _load_commands(self) -> None:
        """Load command history from file."""
        try:
            if self._history_file.exists():
                self._commands = self._history_file.read_text().splitlines()
        except Exception:
            logger.exception("Failed to load command history")
            self._commands = []

    def add_command(self, command: str) -> None:
        """Add command to history."""
        if not command.strip():
            return

        with Session(engine) as session:
            history = CommandHistory(
                session_id=str(self.id),  # Convert UUID to str
                agent_name=self._agent.name,
                command=command,
            )
            session.add(history)
            session.commit()

    def get_commands(
        self, limit: int | None = None, current_session_only: bool = False
    ) -> list[str]:
        """Get command history ordered by newest first."""
        with Session(engine) as session:
            query = select(CommandHistory)
            if current_session_only:
                query = query.where(CommandHistory.session_id == str(self.id))
            else:
                query = query.where(CommandHistory.agent_name == self._agent.name)

            # Use the column reference from the model class
            query = query.order_by(desc(CommandHistory.timestamp))  # type: ignore
            if limit:
                query = query.limit(limit)
            return [h.command for h in session.exec(query)]

    @property
    def metadata(self) -> ChatSessionMetadata:
        """Get current session metadata."""
        return ChatSessionMetadata(
            session_id=self.id,
            agent_name=self._agent.name,
            model=self._model,
            tool_states=self._tool_states,
        )

    async def clear(self) -> None:
        """Clear chat history."""
        self._history = []
        event = HistoryClearedEvent(session_id=str(self.id))
        self.history_cleared.emit(event)

    async def reset(self) -> None:
        """Reset session state."""
        old_tools = self._tool_states.copy()
        self._history = []
        self._tool_states = self._agent.tools.list_tools()

        event = SessionResetEvent(
            session_id=str(self.id),
            previous_tools=old_tools,
            new_tools=self._tool_states,
        )
        self.session_reset.emit(event)

    def register_command(self, command: BaseCommand) -> None:
        """Register additional command."""
        self._command_store.register_command(command)

    async def handle_command(
        self,
        command_str: str,
        output: OutputWriter,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Handle a slash command.

        Args:
            command_str: Command string without leading slash
            output: Output writer implementation
            metadata: Optional interface-specific metadata
        """
        self._ensure_initialized()
        ctx = CommandContext(output=output, session=self, metadata=metadata or {})
        await self._command_store.execute_command(command_str, ctx)

    @overload
    async def send_message(
        self,
        content: str,
        *,
        stream: Literal[False] = False,
        output: OutputWriter | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ChatMessage: ...

    @overload
    async def send_message(
        self,
        content: str,
        *,
        stream: Literal[True],
        output: OutputWriter | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncIterator[ChatMessage]: ...

    async def send_message(
        self,
        content: str,
        *,
        stream: bool = False,
        output: OutputWriter | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ChatMessage | AsyncIterator[ChatMessage]:
        """Send a message and get response(s)."""
        self._ensure_initialized()
        if not content.strip():
            msg = "Message cannot be empty"
            raise ValueError(msg)

        if content.startswith("/"):
            writer = output or DefaultOutputWriter()
            try:
                await self.handle_command(content[1:], output=writer, metadata=metadata)
                return ChatMessage(content="", role="system")
            except ExitCommandError:
                # Re-raise without wrapping in CommandError
                raise
            except CommandError as e:
                return ChatMessage(content=f"Command error: {e}", role="system")

        try:
            # Update tool states in pydantic agent before call
            self._agent._pydantic_agent._function_tools.clear()
            enabled_tools = self._agent.tools.get_tools(state="enabled")
            for tool in enabled_tools:
                assert tool._original_callable
                self._agent._pydantic_agent.tool_plain(tool._original_callable)

            if stream:
                return self._stream_message(content)
            return await self._send_normal(content)

        except Exception as e:
            logger.exception("Error processing message")
            msg = f"Error processing message: {e}"
            raise ChatSessionConfigError(msg) from e

    async def _send_normal(self, content: str) -> ChatMessage:
        """Send message and get single response."""
        model_override = self._model if self._model and self._model.strip() else None

        result = await self._agent.run(
            content,
            message_history=self._history,
            model=model_override,  # type: ignore
        )

        # Update history with new messages
        self._history = result.new_messages()

        model_name = model_override or self._agent.model_name
        response = str(result.data)
        cost_info = (
            await extract_token_usage_and_cost(
                result.usage(),
                model_name,
                content,  # prompt
                response,  # completion
            )
            if model_name
            else None
        )

        metadata = {}
        if cost_info:
            metadata.update({
                "token_usage": cost_info.token_usage,
                "cost_usd": cost_info.cost_usd,
            })
        if model_name:
            metadata["model"] = model_name

        # Update session state before returning
        self._state.message_count += 2  # User and assistant messages
        usage = cost_info.token_usage if cost_info else None
        cost = cost_info.cost_usd if cost_info else None
        metadata_obj = MessageMetadata(model=model_name, token_usage=usage, cost=cost)

        chat_msg = ChatMessage(
            content=response,
            role="assistant",
            metadata=metadata_obj,
            token_usage=metadata_obj.token_usage,
        )
        self._state.update_tokens(chat_msg)

        return chat_msg

    async def _stream_message(
        self,
        content: str,
    ) -> AsyncIterator[ChatMessage]:
        """Send message and stream responses."""
        async with await self._agent.run_stream(
            content,
            message_history=self._history,
            model=self._model or "",  # type: ignore
        ) as stream_result:
            async for response in stream_result.stream():
                chat_msg = ChatMessage(
                    content=str(response),
                    role="assistant",
                )
                yield chat_msg

            # Final message with token usage after stream completes
            model_name = self._model or self._agent.model_name
            cost_info = (
                await extract_token_usage_and_cost(
                    stream_result.usage(),
                    model_name,
                    content,  # prompt
                    response,  # completion
                )
                if model_name
                else None
            )
            metadata = {}
            if cost_info:
                metadata.update({
                    "token_usage": cost_info.token_usage,
                    "cost_usd": cost_info.cost_usd,
                })
            if model_name:
                metadata["model"] = model_name

            # Update session state after stream completes
            self._state.message_count += 2  # User and assistant messages
            meta_obj = MessageMetadata(**metadata)
            final_msg = ChatMessage(
                content="",  # Empty content for final status message
                role="assistant",
                metadata=meta_obj,
                token_usage=meta_obj.token_usage,
            )
            self._state.update_tokens(final_msg)
            yield final_msg

    def configure_tools(
        self,
        updates: dict[str, bool],
    ) -> dict[str, str]:
        """Update tool configuration.

        Args:
            updates: Mapping of tool names to desired states

        Returns:
            Mapping of tool names to status messages
        """
        results = {}
        for tool, enabled in updates.items():
            try:
                if enabled:
                    self._agent.tools.enable_tool(tool)
                    results[tool] = "enabled"
                else:
                    self._agent.tools.disable_tool(tool)
                    results[tool] = "disabled"
                self._tool_states[tool] = enabled
            except ValueError as e:
                results[tool] = f"error: {e}"

        logger.debug("Updated tool states for session %s: %s", self.id, results)
        return results

    def get_tool_states(self) -> dict[str, bool]:
        """Get current tool states."""
        return self._agent.tools.list_tools()

    @property
    def tools(self) -> ToolManager:
        """Get current tool states."""
        return self._agent.tools

    @property
    def history(self) -> list[messages.ModelMessage]:
        """Get conversation history."""
        return list(self._history)

    def get_status(self) -> SessionState:
        """Get current session status."""
        return self._state
