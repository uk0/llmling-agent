"""Web interface commands."""

from __future__ import annotations

import webbrowser

import typer as t

from llmling_agent.web import launch_web_ui


web_cli = t.Typer(help="Web interface commands", no_args_is_help=True)


@web_cli.command("launch")
def launch_gui(
    host: str = t.Option("127.0.0.1", "--host", "-h", help="Host to bind to"),
    port: int = t.Option(None, "--port", "-p", help="Port to bind to"),
    share: bool = t.Option(False, "--share", help="Create public URL"),
    browser: bool = t.Option(True, "--browser/--no-browser", help="Open in browser"),
) -> None:
    """Launch the web interface."""
    if browser:
        url = f"http://{host}:{port or 7860}"
        from threading import Timer

        Timer(1.5, webbrowser.open, args=[url]).start()

    launch_web_ui(
        server_name=host,
        server_port=port,
        share=share,
    )
