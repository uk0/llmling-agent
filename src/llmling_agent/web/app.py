"""Web interface for LLMling agents."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import gradio as gr
from llmling.config.store import ConfigStore
from upath import UPath

from llmling_agent.models import AgentDefinition
from llmling_agent.web.handlers import AgentHandler


if TYPE_CHECKING:
    from pathlib import Path


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


def _get_agent_files() -> list[str]:
    """Get list of available agent configuration files.

    Returns:
        List of agent configuration file paths
    """
    store = ConfigStore("agents.json")
    configs = store.list_configs()
    paths = [str(UPath(cfg[1])) for cfg in configs]

    # Also look in current directory
    current_path = UPath.cwd()
    yaml_files = current_path.glob("*.yml")
    paths.extend(str(p) for p in yaml_files if _is_agent_file(p))

    return sorted(set(paths))


def _is_agent_file(path: Path) -> bool:
    """Check if YAML file contains agent configuration.

    Args:
        path: Path to check

    Returns:
        True if file is a valid agent configuration
    """
    path = UPath(path) if isinstance(path, str) else path
    try:
        AgentDefinition.from_file(str(path))
    except Exception:  # noqa: BLE001
        return False
    else:
        return True


class AgentUI:
    """Main agent web interface."""

    def __init__(self) -> None:
        """Initialize interface."""
        self.handler: AgentHandler | None = None
        self.available_files = _get_agent_files()
        logger.debug("Available files: %s", self.available_files)

    def create_ui(self) -> gr.Blocks:
        """Create the Gradio interface.

        Returns:
            Configured Gradio Blocks interface
        """
        with gr.Blocks(css=CUSTOM_CSS) as app:
            gr.Markdown("# 🤖 LLMling Agent Chat")

            with gr.Row():
                with gr.Column(scale=1):
                    # Agent file selection
                    file_input = gr.Dropdown(
                        choices=self.available_files,
                        label="Agent Configuration File",
                        interactive=True,
                        value=self.available_files[0] if self.available_files else None,
                    )

                    # Agent selection (populated after file load)
                    agent_input = gr.Dropdown(
                        choices=[], label="Select Agent", interactive=True
                    )

                    # Model override
                    model_input = gr.Textbox(
                        label="Model Override (optional)",
                        placeholder="e.g. openai:gpt-4",
                        interactive=True,
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
                        type="messages",  # Use newer message format
                    )

                    with gr.Row():
                        msg_box = gr.Textbox(
                            placeholder="Type your message here...",
                            label="Message",
                            scale=8,
                            container=False,
                        )
                        submit_btn = gr.Button("Send", scale=1, variant="primary")

                    status = gr.Markdown("No agent selected", elem_classes=["status-msg"])

            # Event handlers with explicit debug
            def on_file_selected(file_path: str | None) -> tuple[list[str], str]:
                logger.debug("File selection triggered with: %s", file_path)
                return self._on_file_selected(file_path)

            file_input.change(
                fn=on_file_selected,
                inputs=file_input,
                outputs=[agent_input, status],
            )

            def on_agent_selected(
                agent_name: str | None,
                model: str | None,
            ) -> tuple[str, list[list[str]]]:
                logger.debug(
                    "Agent selection triggered with: %s (model: %s)", agent_name, model
                )
                return self._on_agent_selected(agent_name, model)

            agent_input.change(
                fn=on_agent_selected,
                inputs=[agent_input, model_input],
                outputs=[status, chatbot],
            )

            def on_message_sent(
                message: str,
                history: list[list[str]],
            ) -> tuple[str, list[list[str]], str]:
                logger.debug("Message send triggered: %s", message)
                return self._on_message_sent(message, history)

            submit_btn.click(
                fn=on_message_sent,
                inputs=[msg_box, chatbot],
                outputs=[msg_box, chatbot, status],
            )
            msg_box.submit(
                fn=on_message_sent,
                inputs=[msg_box, chatbot],
                outputs=[msg_box, chatbot, status],
            )

        return app

    async def _on_file_selected(
        self,
        file_path: str | None,
    ) -> tuple[list[str], str]:
        """Handle file selection event.

        Args:
            file_path: Selected file path

        Returns:
            Tuple of (agent choices, status message)
        """
        if not file_path:
            return [], "No file selected"

        logger.debug("Loading agent file: %s", file_path)
        self.handler = AgentHandler(file_path)
        result = await self.handler.load_agent_file(file_path)
        logger.debug("Load result: %s", result)
        return result

    async def _on_agent_selected(
        self,
        agent_name: str | None,
        model: str | None,
    ) -> tuple[str, list[list[str]]]:
        """Handle agent selection event.

        Args:
            agent_name: Selected agent name
            model: Optional model override

        Returns:
            Tuple of (status message, chat history)
        """
        if not self.handler:
            return "No configuration loaded", []
        if not agent_name:
            return "No agent selected", []

        # Pass current file path from handler along with other params
        return await self.handler.select_agent(
            file_path=self.handler._file_path, agent_name=agent_name, model=model
        )

    async def _on_message_sent(
        self,
        message: str,
        chat_history: list[list[str]],
    ) -> tuple[str, list[list[str]], str]:
        """Handle message send event.

        Args:
            message: User message
            chat_history: Current chat history

        Returns:
            Tuple of (cleared message, updated history, status)
        """
        if not self.handler:
            return message, chat_history, "No agent loaded"

        return await self.handler.send_message(message, chat_history)


def create_app() -> gr.Blocks:
    """Create the Gradio web interface.

    Returns:
        Configured Gradio application
    """
    ui = AgentUI()
    return ui.create_ui()


def launch_app(
    *,
    share: bool = False,
    server_name: str = "127.0.0.1",
    server_port: int | None = None,
    auth: tuple[str, str] | None = None,
) -> None:
    """Launch the web interface.

    Args:
        share: Whether to create a public link
        server_name: Server hostname
        server_port: Server port number
        auth: Optional (username, password) for basic auth
    """
    setup_logging()
    logger.debug("Starting web interface")
    app = create_app()
    app.queue()
    app.launch(share=share, server_name=server_name, server_port=server_port, auth=auth)


def setup_logging() -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


if __name__ == "__main__":
    launch_app()
