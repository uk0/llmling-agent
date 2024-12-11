"""Web interface commands."""

from __future__ import annotations

from threading import Timer
import webbrowser

import typer as t

from llmling_agent.web.app import launch_app


def launch_gui(
    host: str = t.Option("127.0.0.1", "--host", "-h", help="Host to bind to"),
    port: int = t.Option(None, "--port", "-p", help="Port to bind to"),
    share: bool = t.Option(False, "--share", help="Create public URL"),
    browser: bool = t.Option(True, "--browser/--no-browser", help="Open in browser"),
) -> None:
    """Launch the web interface."""
    if browser:
        url = f"http://{host}:{port or 7860}"
        Timer(1.5, webbrowser.open, args=[url]).start()
    launch_app(server_name=host, server_port=port, share=share, block=True)
