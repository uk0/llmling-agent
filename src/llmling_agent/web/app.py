"""Web interface for LLMling agents."""

from __future__ import annotations

import logging
from typing import Any

import gradio as gr
from llmling.config.store import ConfigStore
from upath import UPath
import yaml

from llmling_agent.web.handlers import AgentHandler


logger = logging.getLogger(__name__)

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
"""


def load_yaml(path: str) -> dict:
    """Load and parse YAML file."""
    with UPath(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class AgentUI:
    """Main agent web interface."""

    def __init__(self) -> None:
        """Initialize interface."""
        store = ConfigStore("agents.json")
        self.available_files = [str(UPath(path)) for _, path in store.list_configs()]
        logger.debug("Available files: %s", self.available_files)
        self._handler: AgentHandler | None = None
        self.initial_status = "Please select a configuration file"

    @property
    def handler(self) -> AgentHandler:
        """Get the current agent handler.

        Returns:
            AgentHandler: The current handler

        Raises:
            ValueError: If no handler is initialized
        """
        if self._handler is None:
            msg = "No configuration file selected"
            raise ValueError(msg)
        return self._handler

    def create_ui(self) -> gr.Blocks:
        """Create the Gradio interface."""

        def handle_file_selection(evt: gr.SelectData) -> tuple[dict[str, Any], str]:
            """Handle file selection event."""
            file_path = evt.value
            logger.info("File selection event: %s", file_path)
            try:
                # Create new handler with selected file
                self._handler = AgentHandler(file_path)

                data = load_yaml(file_path)
                agents = list(data.get("agents", {}).keys())
                msg = f"Loaded {len(agents)} agents: {', '.join(agents)}"
                logger.info(msg)
                return gr.update(choices=agents, value=None), msg
            except Exception as e:
                logger.exception("Failed to load file")
                self._handler = None
                return gr.update(choices=[], value=None), f"Error: {e}"

        async def handle_agent_selection(
            agent_name: str | None, model: str | None, history: list[dict[str, str]]
        ) -> tuple[str, list[dict[str, str]]]:
            """Handle agent selection."""
            logger.info("Agent selection event: %s (model: %s)", agent_name, model)
            if not agent_name:
                return "No agent selected", history

            try:
                status_msg, list_history = await self.handler.select_agent(
                    file_path=self.handler._file_path, agent_name=agent_name, model=model
                )

                # Convert list[list[str]] to list[dict[str, str]]
                dict_history = [
                    {"role": "user" if i % 2 == 0 else "assistant", "content": msg}
                    for sublist in list_history
                    for i, msg in enumerate(sublist)
                ]
            except ValueError as e:
                return str(e), history
            except Exception as e:
                logger.exception("Failed to initialize agent")
                return f"Error initializing agent: {e}", history
            else:
                return status_msg, dict_history

        async def send_message(
            message: str,
            history: list[dict[str, str]],
            agent_name: str | None,
            model: str | None,
        ) -> tuple[str, list[dict[str, str]], str]:
            """Handle message sending."""
            logger.info(
                "Message send event: %s (agent: %s, model: %s)",
                message,
                agent_name,
                model,
            )

            if not agent_name:
                return message, history, "Please select an agent first"

            if not message.strip():
                return message, history, "Message is empty"

            try:
                assert self.handler.state.current_agent
                result = await self.handler.state.current_agent.run(message)
                response = result.data

                new_history = list(history)
                new_history.extend([
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": response},
                ])
            except ValueError as e:
                return message, history, str(e)
            except Exception as e:
                logger.exception("Failed to process message")
                return message, history, f"Error: {e}"
            else:
                return "", new_history, "Message sent"

        def handle_upload(upload: gr.FileData) -> tuple[list[str], list[str], str]:
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

                return (
                    new_files,
                    agents,
                    f"Loaded {len(agents)} agents from {file_path.name}",
                )
            except Exception as e:
                logger.exception("Failed to upload file")
                return self.available_files, [], f"Error uploading file: {e}"

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
                        avatar_images=("👤", "🤖"),
                        bubble_full_width=False,
                        type="messages",
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

            # Event handlers
            upload_button.upload(
                fn=handle_upload,
                inputs=upload_button,
                outputs=[file_input, agent_input, status],
            )

            file_input.select(
                fn=handle_file_selection,
                inputs=None,
                outputs=[agent_input, status],
            )

            agent_input.select(
                fn=handle_agent_selection,
                inputs=[agent_input, model_input, chatbot],
                outputs=[status, chatbot],
            )

            msg_box.submit(
                fn=send_message,
                inputs=[msg_box, chatbot, agent_input, model_input],
                outputs=[msg_box, chatbot, status],
            )

            submit_btn.click(
                fn=send_message,
                inputs=[msg_box, chatbot, agent_input, model_input],
                outputs=[msg_box, chatbot, status],
            )

        return app


def setup_logging() -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        force=True,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logging.getLogger("gradio").setLevel(logging.INFO)
    logging.getLogger("llmling_agent").setLevel(logging.INFO)


def create_app() -> gr.Blocks:
    """Create the Gradio interface."""
    ui = AgentUI()
    return ui.create_ui()


def launch_app(
    *,
    share: bool = False,
    server_name: str = "127.0.0.1",
    server_port: int | None = None,
) -> None:
    """Launch the web interface."""
    setup_logging()
    logger.info("Starting web interface")
    app = create_app()
    app.queue()
    app.launch(
        share=share,
        server_name=server_name,
        server_port=server_port,
        prevent_thread_lock=__name__ != "__main__",
    )


if __name__ == "__main__":
    launch_app()
