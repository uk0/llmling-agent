"""Web interface for LLMling agents."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal, NotRequired, TypedDict

import gradio as gr
from llmling.config.store import ConfigStore
from upath import UPath

from llmling_agent.web.ui_state import UIState


if TYPE_CHECKING:
    from gradio.routes import App


type ChatHistory = list[ChatMessage]

logger = logging.getLogger(__name__)


class ChatMessage(TypedDict):
    """Single chat message format for Gradio chatbot."""

    content: str
    role: Literal["user", "assistant"]
    name: NotRequired[str]
    avatar: NotRequired[str]


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
) -> tuple[App, str, str]:
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
    return app.launch(
        share=share,
        server_name=server_name,
        server_port=server_port,
        prevent_thread_lock=not block,
    )


if __name__ == "__main__":
    launch_app()
