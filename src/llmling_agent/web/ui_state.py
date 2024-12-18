"""UI state management for web interface."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import gradio as gr
from llmling.config.store import ConfigStore
from pydantic import BaseModel, model_validator
from upath import UPath
import yamling

from llmling_agent.chat_session import AgentChatSession, ChatSessionManager
from llmling_agent.chat_session.events import (
    SessionEvent,
    SessionEventHandler,
    SessionEventType,
)
from llmling_agent.chat_session.models import ChatMessage
from llmling_agent.commands.base import OutputWriter
from llmling_agent.log import LogCapturer
from llmling_agent.web.handlers import AgentHandler
from llmling_agent.web.type_utils import ChatHistory, validate_chat_message


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable


logger = logging.getLogger(__name__)


class WebEventHandler(SessionEventHandler):
    """Handles session events for web interface."""

    def __init__(self, ui_state: UIState) -> None:
        self.ui_state = ui_state

    async def handle_session_event(self, event: SessionEvent) -> None:
        match event.type:
            case SessionEventType.HISTORY_CLEARED:
                # Use empty history with existing UI update mechanism
                await self.ui_state.send_message(
                    message="",
                    history=[],
                    agent_name=None,
                    model=None,
                )
            case SessionEventType.SESSION_RESET:
                # Clear chat and update tool states
                _update = await self.ui_state.send_message(
                    message="",
                    history=[],
                    agent_name=None,
                    model=None,
                )
                await self.ui_state.update_tool_states(event.data["new_tools"])


class WebOutputWriter(OutputWriter):
    """Output writer that sends messages to web UI."""

    def __init__(
        self,
        message_callback: Callable[[ChatMessage], Awaitable[None]],
    ) -> None:
        """Initialize web output writer.

        Args:
            message_callback: Async function to call with new messages
        """
        self._callback = message_callback

    async def print(self, message: str) -> None:
        """Send message to web UI."""
        logger.debug("WebOutputWriter printing: %s", message)
        chat_message = ChatMessage(content=message, role="system")
        await self._callback(chat_message)


class UIUpdate(BaseModel):
    """State updates for the UI components."""

    message_box: str | None = None
    chat_history: ChatHistory | None = None
    status: str | None = None
    debug_logs: str | None = None
    agent_choices: list[str] | None = None
    file_choices: list[str] | None = None
    tool_states: list[list[Any]] | None = None

    @model_validator(mode="after")
    def validate_chat_history(self) -> UIUpdate:
        """Validate chat history format."""
        if self.chat_history is not None:
            for msg in self.chat_history:
                validate_chat_message(msg)
        return self

    def to_updates(self, outputs: list[gr.components.Component]) -> list[Any]:
        """Convert to list of gradio updates matching output components."""
        updates = []
        for output in outputs:
            match output:
                case gr.Dropdown():
                    if self.agent_choices is not None:
                        updates.append(gr.update(choices=self.agent_choices))
                    elif self.file_choices is not None:
                        updates.append(gr.update(choices=self.file_choices))
                    else:
                        updates.append(gr.update())
                case gr.Markdown():
                    if output.elem_classes and "debug-logs" in output.elem_classes:
                        updates.append(gr.update(value=self.debug_logs))
                    else:
                        updates.append(gr.update(value=self.status))
                case gr.Chatbot():
                    updates.append(gr.update(value=self.chat_history))
                case gr.Textbox():
                    updates.append(gr.update(value=self.message_box))
                case gr.Dataframe():
                    updates.append(gr.update(value=self.tool_states))
                case _:
                    updates.append(gr.update())
        return updates


class UIState:
    """Maintains UI state and handles updates."""

    def __init__(self) -> None:
        """Initialize UI state."""
        self.log_capturer = LogCapturer()
        self.debug_mode = False
        self.handler: AgentHandler | None = None
        self._event_handler: WebEventHandler | None = None
        self._session_manager = ChatSessionManager()
        self._current_session: AgentChatSession | None = None

    def toggle_debug(self, enabled: bool) -> UIUpdate:
        """Toggle debug mode."""
        self.debug_mode = enabled
        if enabled:
            self.log_capturer.start()
            msg = "Debug mode enabled. Logs will appear here."
            return UIUpdate(debug_logs=msg, status="Debug mode enabled")
        self.log_capturer.stop()
        return UIUpdate(debug_logs=None, status="Debug mode disabled")

    def get_debug_logs(self) -> str | None:
        """Get current debug logs if debug mode is enabled."""
        return self.log_capturer.get_logs() if self.debug_mode else None

    async def handle_file_selection(self, file_path: str) -> UIUpdate:
        """Handle file selection event."""
        logger.info("File selection event: %s", file_path)
        try:
            self.handler = await AgentHandler.create(file_path)
            agents = list(self.handler.state.agent_def.agents)
            msg = f"Loaded {len(agents)} agents: {', '.join(agents)}"
            logger.info(msg)
            logs = self.get_debug_logs()
            return UIUpdate(agent_choices=agents, status=msg, debug_logs=logs)
        except Exception as e:
            logger.exception("Failed to load file")
            self.handler = None
            logs = self.get_debug_logs()
            return UIUpdate(agent_choices=[], status=f"Error: {e}", debug_logs=logs)

    async def handle_upload(self, upload: gr.FileData) -> UIUpdate:
        """Handle config file upload."""
        try:
            # Save file to configs directory
            config_dir = UPath("configs")
            config_dir.mkdir(exist_ok=True)
            assert upload.orig_name
            file_path = config_dir / upload.orig_name
            UPath(upload.path).rename(file_path)
            logger.info("Saved config to: %s", file_path)

            # Add to store
            name = file_path.stem
            store = ConfigStore("agents.json")
            store.add_config(name, str(file_path))

            # Update available files and load agents
            files = [str(UPath(p)) for _, p in store.list_configs()]
            data = yamling.load_yaml_file(str(file_path))
            agents = list(data.get("agents", {}).keys())
            msg = f"Loaded {len(agents)} agents from {file_path.name}"
            logs = self.get_debug_logs()
            return UIUpdate(
                file_choices=files, agent_choices=agents, status=msg, debug_logs=logs
            )
        except Exception as e:
            logger.exception("Failed to upload file")
            status = f"Error uploading file: {e}"
            return UIUpdate(status=status, debug_logs=self.get_debug_logs())

    async def handle_agent_selection(
        self,
        agent_name: str | None,
        model: str | None,
        history: ChatHistory,
    ) -> UIUpdate:
        """Handle agent selection."""
        if not agent_name:
            return UIUpdate(status="No agent selected", chat_history=history)

        try:
            if not self.handler:
                msg = "No configuration loaded"
                raise ValueError(msg)  # noqa: TRY301

            # Initialize the runner
            await self.handler.select_agent(agent_name, model)

            # Get the agent from the runner
            if not self.handler.state.current_runner:
                msg = f"Failed to initialize runner for {agent_name}"
                raise ValueError(msg)  # noqa: TRY301

            agent = self.handler.state.current_runner.agent

            # Create chat session with the agent
            if self._current_session and self._event_handler:
                self._current_session.remove_event_handler(self._event_handler)

            # Create new session
            self._current_session = await self._session_manager.create_session(
                agent=agent,
                model=model,
            )

            # Register new event handler
            self._event_handler = WebEventHandler(self)
            self._current_session.add_event_handler(self._event_handler)
            # Get tool states for UI
            states = self._current_session.get_tool_states()
            tool_states = [[name, enabled] for name, enabled in states.items()]

            return UIUpdate(
                status=f"Agent {agent_name} ready",
                chat_history=[],  # type: ignore
                tool_states=tool_states,
                debug_logs=self.get_debug_logs(),
            )

        except Exception as e:
            logger.exception("Failed to initialize agent")
            logs = self.get_debug_logs()
            return UIUpdate(status=f"Error: {e}", debug_logs=logs)

    async def send_message(
        self,
        message: str,
        history: ChatHistory,
        agent_name: str | None,
        model: str | None,
    ) -> UIUpdate:
        """Handle message sending."""
        if not message.strip():
            logs = self.get_debug_logs()
            return UIUpdate(message_box="", status="Message is empty", debug_logs=logs)

        if not self._current_session:
            return UIUpdate(
                message_box=message,
                chat_history=history,
                status="No active session",
                debug_logs=self.get_debug_logs(),
            )

        try:
            messages = list(history)

            # For commands, add the command as user message
            if message.startswith("/"):
                messages.append({"content": message, "role": "user"})

            # Collect command outputs
            command_outputs: list[str] = []

            async def add_message(msg: ChatMessage) -> None:
                if message.startswith("/"):
                    command_outputs.append(msg.content)
                else:
                    messages.append({"content": msg.content, "role": msg.role})

            # Send message through chat session
            writer = WebOutputWriter(add_message)
            result = await self._current_session.send_message(message, output=writer)

            # For non-command messages, add the regular response
            if not message.startswith("/"):
                messages.append({"content": message, "role": "user"})
                if result.content:
                    messages.append({"content": result.content, "role": "assistant"})
            # For commands, add the collected output as a response
            elif command_outputs:
                content = "\n".join(command_outputs)
                messages.append({"content": content, "role": "assistant"})

            logger.debug("Final messages: %s", messages)
            return UIUpdate(
                message_box="",
                chat_history=messages,
                status="Message sent",
                debug_logs=self.get_debug_logs(),
            )

        except Exception as e:
            logger.exception("Failed to process message")
            return UIUpdate(
                message_box=message,
                chat_history=history,
                status=f"Error: {e}",
                debug_logs=self.get_debug_logs(),
            )

    async def stream_message(
        self,
        message: str,
        history: ChatHistory,
    ) -> AsyncIterator[UIUpdate]:
        """Stream message responses."""
        if not self._current_session:
            yield UIUpdate(status="No active session")
            return

        try:
            messages = list(history)
            messages.append({"content": message, "role": "user"})

            # Get the iterator from send_message
            message_iterator = await self._current_session.send_message(
                message, stream=True
            )

            # Iterate over the chat messages
            async for chat_msg in message_iterator:
                messages.append({
                    "content": chat_msg.content,
                    "role": "assistant",
                    "metadata": chat_msg.metadata or {},
                })

                yield UIUpdate(chat_history=messages, status="Receiving response...")

            # Final update
            yield UIUpdate(
                message_box="",
                chat_history=messages,
                status="Message sent",
                debug_logs=self.get_debug_logs(),
            )

        except Exception as e:
            logger.exception("Failed to stream message")
            logs = self.get_debug_logs()
            yield UIUpdate(status=f"Error: {e}", debug_logs=logs)

    async def update_tool_states(self, updates: dict[str, bool]) -> UIUpdate:
        """Update tool states in current session."""
        if not self._current_session:
            return UIUpdate(status="No active session")

        try:
            results = self._current_session.configure_tools(updates)
            status = "; ".join(f"{k}: {v}" for k, v in results.items())

            # Get updated tool states
            manager = self._current_session.tools
            tool_states = [[t.name, t.enabled] for t in manager.values()]
            logs = self.get_debug_logs()
            msg = f"Updated tools: {status}"
            return UIUpdate(status=msg, tool_states=tool_states, debug_logs=logs)

        except Exception as e:
            logger.exception("Failed to update tools")
            logs = self.get_debug_logs()
            return UIUpdate(status=f"Error updating tools: {e}", debug_logs=logs)
