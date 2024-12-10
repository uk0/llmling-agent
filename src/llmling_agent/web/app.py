"""Web interface for LLMling agents."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Literal, NotRequired, TypedDict

import gradio as gr
from llmling.config.store import ConfigStore
from pydantic import BaseModel
from pydantic_ai import messages
from upath import UPath
import yaml

from llmling_agent.log import LogCapturer
from llmling_agent.web.handlers import AgentHandler


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


class ChatMessage(TypedDict):
    """Single chat message format for Gradio chatbot."""

    content: str
    role: Literal["user", "assistant"]
    name: NotRequired[str]
    avatar: NotRequired[str]


type ChatHistory = list[ChatMessage]


CUSTOM_CSS = """
.agent-chat {
    height: 600px;
    border-radius: 8px;
    border: 1px solid #e0e0e0;
    background: #ffffff;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.agent-chat .message.user {
    background: #f0f7ff;
    border: 1px solid #e1effe;
}

.agent-chat .message.bot {
    background: #f8f9fa;
    border: 1px solid #e9ecef;
}

.status-msg {
    text-align: center;
    color: #666;
    padding: 8px;
    border-radius: 4px;
    background: #f8f9fa;
    margin: 8px 0;
}

.debug-logs {
    font-family: monospace;
    white-space: pre-wrap;
    background: #f8f9fa;
    padding: 8px;
    border-radius: 4px;
    border: 1px solid #e9ecef;
    margin-top: 8px;
    font-size: 0.9em;
    max-height: 200px;
    overflow-y: auto;
}
"""


def load_yaml(path: str) -> dict[str, Any]:
    """Load and parse YAML file."""
    with UPath(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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
            return UIUpdate(
                debug_logs="Debug mode enabled. Logs will appear here.",
                status="Debug mode enabled",
            )
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
            data = load_yaml(str(file_path))
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
            return UIUpdate(
                status=f"Error uploading file: {e}",
                debug_logs=self.get_debug_logs(),
            )

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
            return UIUpdate(
                status=f"Agent {agent_name} ready",
                chat_history=agent_history,
                debug_logs=self.get_debug_logs(),
            )
        except Exception as e:
            logger.exception("Failed to initialize agent")
            return UIUpdate(
                status=f"Error: {e}",
                debug_logs=self.get_debug_logs(),
            )

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


class AgentUI:
    """Main agent web interface."""

    def __init__(self) -> None:
        """Initialize interface."""
        store = ConfigStore("agents.json")
        self.available_files = [str(UPath(path)) for _, path in store.list_configs()]
        self.state = UIState()
        self.initial_status = "Please select a configuration file"

    def create_ui(self) -> gr.Blocks:
        """Create the Gradio interface."""
        with gr.Blocks(css=CUSTOM_CSS) as app:
            gr.Markdown("# 🤖 LLMling Agent Chat")

            with gr.Row():
                with gr.Column(scale=1):
                    # Config file management
                    with gr.Group(visible=True):
                        upload_button = gr.UploadButton(
                            "📁 Upload Config",
                            file_types=[".yml", ".yaml"],
                            file_count="single",
                            interactive=True,
                        )
                        file_input = gr.Dropdown(
                            choices=self.available_files,
                            label="Agent Configuration File",
                            value=None,  # No default selection
                            interactive=True,
                            show_label=True,
                        )

                    # Agent selection - empty initially
                    agent_input = gr.Dropdown(
                        choices=[],
                        label="Select Agent",
                        interactive=True,
                        show_label=True,
                    )

                    status = gr.Markdown(
                        value=self.initial_status,
                        elem_classes=["status-msg"],
                    )

                    # Model override
                    model_input = gr.Textbox(
                        label="Model Override (optional)",
                        placeholder="e.g. openai:gpt-4",
                        interactive=True,
                        show_label=True,
                    )

                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(
                        value=[],
                        label="Chat",
                        height=600,
                        show_copy_button=True,
                        show_copy_all_button=True,
                        type="messages",  # Important: specify messages type
                        avatar_images=("👤", "🤖"),
                        bubble_full_width=False,
                    )

                    with gr.Row():
                        msg_box = gr.Textbox(
                            placeholder="Type your message here...",
                            label="Message",
                            scale=8,
                            container=False,
                            interactive=True,
                        )
                        submit_btn = gr.Button(
                            "Send",
                            scale=1,
                            variant="primary",
                            interactive=True,
                        )

            with gr.Row():
                debug_toggle = gr.Checkbox(
                    label="Debug Mode",
                    value=False,
                    interactive=True,
                )
                debug_logs = gr.Markdown(
                    value=None,
                    visible=True,
                    elem_classes=["debug-logs"],
                )

            # Event handlers with proper async handling
            async def handle_upload(x: Any) -> list[Any]:
                result = await self.state.handle_upload(x)
                return result.to_updates([file_input, agent_input, status, debug_logs])

            async def handle_file_selection(file_path: str) -> list[Any]:
                result = await self.state.handle_file_selection(file_path)
                return result.to_updates([agent_input, status, debug_logs])

            async def handle_agent_selection(*args: Any) -> list[Any]:
                result = await self.state.handle_agent_selection(*args)
                return result.to_updates([status, chatbot, debug_logs])

            def handle_debug(x: bool) -> list[Any]:
                result = self.state.toggle_debug(x)
                return result.to_updates([debug_logs, status])

            async def handle_message(*args: Any) -> list[Any]:
                result = await self.state.send_message(*args)
                return result.to_updates([msg_box, chatbot, status, debug_logs])

            # Connect handlers to UI events
            upload_button.upload(
                fn=handle_upload,
                inputs=[upload_button],
                outputs=[file_input, agent_input, status, debug_logs],
            )

            file_input.select(
                fn=handle_file_selection,
                inputs=[file_input],
                outputs=[agent_input, status, debug_logs],
            )

            agent_input.select(
                fn=handle_agent_selection,
                inputs=[agent_input, model_input, chatbot],
                outputs=[status, chatbot, debug_logs],
            )

            debug_toggle.change(
                fn=handle_debug,
                inputs=[debug_toggle],
                outputs=[debug_logs, status],
            )

            inputs = [msg_box, chatbot, agent_input, model_input]
            outputs = [msg_box, chatbot, status, debug_logs]
            msg_box.submit(fn=handle_message, inputs=inputs, outputs=outputs)
            submit_btn.click(fn=handle_message, inputs=inputs, outputs=outputs)

        return app


def setup_logging() -> None:
    """Set up logging configuration."""
    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, force=True, format=fmt)
    logging.getLogger("gradio").setLevel(logging.INFO)
    logging.getLogger("llmling_agent").setLevel(logging.DEBUG)
    logging.getLogger("llmling").setLevel(logging.DEBUG)


def create_app() -> gr.Blocks:
    """Create the Gradio interface."""
    ui = AgentUI()
    return ui.create_ui()


def launch_app(
    *,
    share: bool = False,
    server_name: str = "127.0.0.1",
    server_port: int | None = None,
    block: bool = True,
) -> None:
    """Launch the LLMling web interface.

    This provides a user-friendly interface to:
    - Load agent configuration files
    - Select and configure agents
    - Chat with agents
    - View chat history and debug logs

    Args:
        share: Whether to create a public URL
        server_name: Server hostname (default: "127.0.0.1")
        server_port: Optional server port number
        block: Whether to block the thread. Set to False when using programmatically.

    Example:
        ```python
        from llmling_agent.web import launch_app

        # Basic local launch
        launch_app()

        # Launch public instance
        launch_app(share=True)

        # Custom server configuration
        launch_app(server_name="0.0.0.0", server_port=8080)

        # Non-blocking for programmatic use
        launch_app(block=False)
        # ... do other things while server runs
        ```
    """
    setup_logging()
    logger.info("Starting web interface")
    app = create_app()
    app.queue()
    app.launch(
        share=share,
        server_name=server_name,
        server_port=server_port,
        prevent_thread_lock=not block,
    )


if __name__ == "__main__":
    launch_app()
