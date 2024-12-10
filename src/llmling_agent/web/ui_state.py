from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import TYPE_CHECKING, Any

import gradio as gr
from llmling.config.store import ConfigStore
from pydantic import BaseModel
from pydantic_ai import messages
from upath import UPath
import yamling

from llmling_agent.log import LogCapturer
from llmling_agent.web.handlers import AgentHandler


if TYPE_CHECKING:
    from llmling_agent.web.app import ChatHistory


logger = logging.getLogger(__name__)


def convert_chat_history_to_messages(
    history: ChatHistory,
) -> list[messages.Message]:
    """Convert Gradio chat history to pydantic-ai messages."""
    result: list[messages.Message] = []
    for msg in history:
        if msg["role"] == "user":
            result.append(messages.UserPrompt(content=msg["content"]))
        elif msg["role"] == "assistant":
            result.append(messages.ModelTextResponse(content=msg["content"]))
    return result


class UIUpdate(BaseModel):
    """State updates for the UI components."""

    message_box: str | None = None
    chat_history: ChatHistory | None = None
    status: str | None = None
    debug_logs: str | None = None
    agent_choices: list[str] | None = None
    file_choices: list[str] | None = None
    selected_agent: str | None = None

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
                case _:
                    updates.append(gr.update())
        return updates


@dataclass
class UIState:
    """Maintains UI state and handles updates."""

    log_capturer: LogCapturer = field(default_factory=LogCapturer)
    debug_mode: bool = False
    handler: AgentHandler | None = None

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
            return UIUpdate(
                agent_choices=agents,
                selected_agent=None,
                status=msg,
                debug_logs=self.get_debug_logs(),
            )
        except Exception as e:
            logger.exception("Failed to load file")
            self.handler = None
            return UIUpdate(
                agent_choices=[],
                selected_agent=None,
                status=f"Error: {e}",
                debug_logs=self.get_debug_logs(),
            )

    async def handle_upload(
        self,
        upload: gr.FileData,
    ) -> UIUpdate:
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
            new_files = [str(UPath(p)) for _, p in store.list_configs()]
            data = yamling.load_yaml_file(str(file_path))
            agents = list(data.get("agents", {}).keys())
            msg = f"Loaded {len(agents)} agents from {file_path.name}"

            return UIUpdate(
                file_choices=new_files,
                agent_choices=agents,
                status=msg,
                debug_logs=self.get_debug_logs(),
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

            await self.handler.select_agent(agent_name, model)
            # Get history for this agent if any
            agent_history = self.handler.state.history.get(agent_name, [])
            logs = self.get_debug_logs()
            status = f"Agent {agent_name} ready"
            return UIUpdate(status=status, chat_history=agent_history, debug_logs=logs)
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
            return UIUpdate(
                message_box="",
                status="Message is empty",
                debug_logs=self.get_debug_logs(),
            )

        if not self.handler or not agent_name:
            return UIUpdate(
                message_box=message,
                chat_history=history,
                status="No agent selected",
                debug_logs=self.get_debug_logs(),
            )

        try:
            runner = self.handler.state.current_runner
            if not runner:
                msg = "Agent not initialized"
                raise ValueError(msg)  # noqa: TRY301

            # Convert to pydantic-ai messages
            msg_history = convert_chat_history_to_messages(history)

            # Get response from agent
            result = await runner.agent.run(message, message_history=msg_history)
            response = str(result.data)

            # Update chat history with new Gradio messages
            new_history = list(history)
            new_history.extend([
                {"role": "user", "content": message},
                {"role": "assistant", "content": response},
            ])

            # Store history
            self.handler.state.history[agent_name] = new_history

            return UIUpdate(
                message_box="",
                chat_history=new_history,
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
