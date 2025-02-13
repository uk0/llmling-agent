"""UI state management for web interface."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import gradio as gr
from llmling import ConfigStore
from pydantic import BaseModel, Field
from slashed import CommandStore
from upath import UPath
import yamling

from llmling_agent.utils.tasks import TaskManagerMixin
from llmling_agent_web.handlers import AgentHandler
from llmling_agent_web.log_capturer import LogCapturer
from llmling_agent_web.type_utils import ChatHistory  # noqa: TC001


if TYPE_CHECKING:
    from llmling_agent import Agent
    from llmling_agent.tools.base import Tool


logger = logging.getLogger(__name__)


class UIUpdate(BaseModel):
    """State updates for the UI components."""

    message_box: str | None = None
    chat_history: ChatHistory = Field(default_factory=list)
    status: str | None = None
    debug_logs: str | None = None
    agent_choices: list[str] | None = None
    file_choices: list[str] | None = None
    tool_states: list[tuple[str, bool]] = Field(default_factory=list)

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


class UIState(TaskManagerMixin):
    """Maintains UI state and handles updates."""

    def __init__(self):
        """Initialize UI state."""
        self.log_capturer = LogCapturer()
        self.debug_mode = False
        self.handler: AgentHandler | None = None
        self._agent: Agent[Any] | None = None
        self.context: Any = None
        self.store = CommandStore()

    def _connect_signals(self):
        """Connect to agent signals."""
        assert self._agent is not None

        # Connect tool events
        self._agent.tools.events.added.connect(self._handle_tool_added)
        self._agent.tools.events.removed.connect(self._handle_tool_removed)
        self._agent.tools.events.changed.connect(self._handle_tool_changed)

    def _handle_tool_added(self, name: str, tool: Tool):
        """Handle tool addition."""
        self.create_task(self.update_tool_states({name: tool.enabled}))

    def _handle_tool_removed(self, name: str):
        """Handle tool removal."""
        self.create_task(self.update_tool_states({}))

    def _handle_tool_changed(self, name: str, tool: Tool):
        """Handle tool state changes."""
        self.create_task(self.update_tool_states({name: tool.enabled}))

    async def cleanup(self):
        """Clean up pending tasks."""
        await self.cleanup_tasks()
        if self.handler:
            await self.handler.cleanup()
            self.handler = None

    async def reset_session(self) -> UIUpdate:
        """Reset session state."""
        if not self._agent:
            return UIUpdate(status="No active session")

        # Reset tool states
        self._agent.tools.reset_states()
        tool_states = [(t.name, t.enabled) for t in self._agent.tools.values()]

        # Clear conversation
        self._agent.conversation.clear()
        logs = self.get_debug_logs()
        return UIUpdate(tool_states=tool_states, status="Session reset", debug_logs=logs)

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
            data = yamling.load_yaml_file(
                str(file_path),
                verify_type=dict,  # type: ignore
                resolve_inherit=True,
            )
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

            # Get agent from pool through handler
            self._agent = await self.handler.select_agent(agent_name, model)

            self.context = self._agent.context

            # Connect signals
            self._connect_signals()

            # Get tool states for UI
            states = [(t.name, t.enabled) for t in self._agent.tools.values()]
            msg = f"Agent {agent_name} ready"
            logs = self.get_debug_logs()
            return UIUpdate(status=msg, tool_states=states, debug_logs=logs)

        except Exception as e:
            logger.exception("Failed to initialize agent")
            self._agent = None
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

        if not self._agent:
            return UIUpdate(
                message_box=message,
                chat_history=history,
                status="No active agent",
                debug_logs=self.get_debug_logs(),
            )

        try:
            messages = list(history)
            messages.append({"content": message, "role": "user"})

            # TODO: commands here
            result = await self._agent.run(message)
            messages.append({"content": str(result.content), "role": "assistant"})

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

    async def update_tool_states(self, updates: dict[str, bool]) -> UIUpdate:
        """Update tool states."""
        if not self._agent:
            return UIUpdate(status="No active agent")

        try:
            results = {}
            for tool, enabled in updates.items():
                try:
                    if enabled:
                        self._agent.tools.enable_tool(tool)
                        results[tool] = "enabled"
                    else:
                        self._agent.tools.disable_tool(tool)
                        results[tool] = "disabled"
                except ValueError as e:
                    results[tool] = f"error: {e}"

            status = "; ".join(f"{k}: {v}" for k, v in results.items())
            tool_states = [(t.name, t.enabled) for t in self._agent.tools.values()]
            logs = self.get_debug_logs()
            msg = f"Updated tools: {status}"
            return UIUpdate(status=msg, tool_states=tool_states, debug_logs=logs)

        except Exception as e:
            logger.exception("Failed to update tools")
            msg = f"Error updating tools: {e}"
            return UIUpdate(status=msg, debug_logs=self.get_debug_logs())
