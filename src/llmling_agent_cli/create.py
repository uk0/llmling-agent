"""Web interface commands."""

from __future__ import annotations

import typer


def create(
    output: str = typer.Option(
        None,
        "-o",
        "--output",
        help="Output file path. If not provided, only displays the config.",
    ),
    add_to_store: bool = typer.Option(
        False, "-a", "--add-to-store", help="Add generated config to ConfigStore"
    ),
    model: str = typer.Option(
        "gpt-4", "-m", "--model", help="Model to use for generation"
    ),
    provider: str = typer.Option(
        "pydantic_ai", "-p", "--provider", help="Provider to use (pydantic_ai or litellm)"
    ),
):
    """Interactive config generator for agents and teams."""
    from llmling_textual.creator.creator_app import ConfigGeneratorApp

    app = ConfigGeneratorApp(model=model, provider=provider)  # type: ignore
    app.run()
