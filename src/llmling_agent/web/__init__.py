from __future__ import annotations

from llmling_agent.web.app import launch_app


def launch_web_ui(
    *,
    share: bool = False,
    server_name: str = "0.0.0.0",
    server_port: int | None = None,
    auth: tuple[str, str] | None = None,
) -> None:
    """Launch the web interface for LLMling agents.

    This provides a user-friendly interface to:
    - Load agent configuration files
    - Select and configure agents
    - Chat with agents
    - View chat history

    Args:
        share: Whether to create a public URL
        server_name: Server hostname (default: "0.0.0.0")
        server_port: Optional server port number
        auth: Optional (username, password) tuple for authentication

    Example:
        ```python
        from llmling_agent import launch_web_ui

        # Basic local launch
        launch_web_ui()

        # Launch with authentication
        launch_web_ui(auth=("admin", "secret123"))

        # Launch public instance
        launch_web_ui(share=True)
        ```
    """
    launch_app(
        share=share,
        server_name=server_name,
        server_port=server_port,
        auth=auth,
    )
