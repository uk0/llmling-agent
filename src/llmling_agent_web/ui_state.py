"""UI state management for web interface."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

import gradio as gr
from llmling.config.store import ConfigStore
from pydantic import BaseModel, model_validator
from upath import UPath
import yamling

from llmling_agent.chat_session import AgentChatSession, ChatSessionManager
from llmling_agent.chat_session.output import CallbackOutputWriter
from llmling_agent.log import LogCapturer
from llmling_agent_web.handlers import AgentHandler
from llmling_agent_web.type_utils import ChatHistory, validate_chat_message


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from llmling_agent.chat_session.events import (
        HistoryClearedEvent,
        SessionResetEvent,
    )
    from llmling_agent.chat_session.models import ChatMessage
    from llmling_agent.tools.base import ToolInfo


logger = logging.getLogger(__name__)


class UIUpdate(BaseModel):
    """State updates for the UI components."""

    message_box: str | None = None
    chat_history: ChatHistory | None = None
    status: str | None = None
    debug_logs: str | None = None
    agent_choices: list[str] | None = None
    file_choices: list[str] | None = None
    tool_states: list[tuple[str, bool]] | None = None

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
                case gr.Dropdown() if self.agent_choices is not None:
                    updates.append(gr.update(choices=self.agent_choices))
                case gr.Dropdown() if self.file_choices is not None:
                    updates.append(gr.update(choices=self.file_choices))
                case gr.Dropdown():
                    updates.append(gr.update())
                case gr.Markdown() if "debug-logs" in (output.elem_classes or []):
                    updates.append(gr.update(value=self.debug_logs))
                case gr.Markdown():
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

    def __init__(self):
        """Initialize UI state."""
        self.log_capturer = LogCapturer()
        self.debug_mode = False
        self.handler: AgentHandler | None = None
        self._session_manager = ChatSessionManager()
        self._current_session: AgentChatSession | None = None
        self._pending_tasks: set[asyncio.Task[Any]] = set()

    def _connect_signals(self):
        """Connect to chat session signals."""
        assert self._current_session is not None

        # Connect to signals with bound methods
        self._current_session.history_cleared.connect(self._on_history_cleared)
        self._current_session.session_reset.connect(self._on_session_reset)
        # Tool events
        self._current_session.tool_added.connect(self._handle_tool_added)
        self._current_session.tool_removed.connect(self._handle_tool_removed)
        self._current_session.tool_changed.connect(self._handle_tool_changed)

    def _handle_tool_added(self, tool: ToolInfo):
        """Sync handler for tool addition."""
        task = asyncio.create_task(self.update_tool_states({tool.name: tool.enabled}))
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)

    def _handle_tool_removed(self, tool_name: str):
        """Sync handler for tool removal."""
        task = asyncio.create_task(self.update_tool_states({}))
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)

    def _handle_tool_changed(self, name: str, tool: ToolInfo):
        """Sync handler for tool state changes."""
        task = asyncio.create_task(self.update_tool_states({name: tool.enabled}))
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)

    async def cleanup(self):
        """Clean up pending tasks."""
        if self._pending_tasks:
            await asyncio.gather(*self._pending_tasks, return_exceptions=True)
            self._pending_tasks.clear()

    async def _on_history_cleared(self, event: HistoryClearedEvent):
        """Handle history cleared event."""
        await self.send_message(message="", history=[], agent_name=None, model=None)

    async def _on_session_reset(self, event: SessionResetEvent):
        """Handle session reset event."""
        # Clear chat and update tool states
        _update = await self.send_message(
            message="", history=[], agent_name=None, model=None
        )
        await self.update_tool_states(event.new_tools)

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
            data = yamling.load_yaml_file(str(file_path), verify_type=dict)
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
            if not self.handler.state.pool:
                msg = f"Failed to initialize pool for {agent_name}"
                raise ValueError(msg)  # noqa: TRY301

            agent = self.handler.state.pool.get_agent(agent_name)
            # Create new session
            manager = self._session_manager
            self._current_session = await manager.create_session(agent=agent)
            self._connect_signals()

            # Get tool states for UI
            tools = self._current_session.tools
            tool_states = [(t.name, t.enabled) for t in tools.values()]

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

            async def add_message(msg: ChatMessage):
                if message.startswith("/"):
                    command_outputs.append(msg.content)
                else:
                    messages.append({"content": msg.content, "role": msg.role})

            # Send message through chat session
            writer = CallbackOutputWriter(add_message)
            result = await self._current_session.send_message(message, output=writer)

            # For non-command messages, add the regular response
            if not message.startswith("/"):
                messages.append({"content": message, "role": "user"})
                if result.content:
                    messages.append({"content": str(result.content), "role": "assistant"})
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
        session = self._current_session
        if not session:
            yield UIUpdate(status="No active session")
            return

        try:
            messages = list(history)
            messages.append({"content": message, "role": "user"})

            # Get the iterator from send_message
            message_iterator = await session.send_message(message, stream=True)

            # Iterate over the chat messages
            async for chat_msg in message_iterator:
                messages.append({
                    "content": str(chat_msg.content),
                    "role": "assistant",
                    "metadata": chat_msg.metadata,
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
            tool_states = [(t.name, t.enabled) for t in manager.values()]
            logs = self.get_debug_logs()
            msg = f"Updated tools: {status}"
            return UIUpdate(status=msg, tool_states=tool_states, debug_logs=logs)

        except Exception as e:
            logger.exception("Failed to update tools")
            logs = self.get_debug_logs()
            return UIUpdate(status=f"Error updating tools: {e}", debug_logs=logs)
