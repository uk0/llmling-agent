"""Web interface commands."""

from __future__ import annotations

import typer

from llmling_agent_cli.cli_types import Provider  # noqa: TC001


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
        "gpt-5", "-m", "--model", help="Model to use for generation"
    ),
    provider: Provider = typer.Option(  # noqa: B008
        "pydantic_ai", "-p", "--provider", help="Provider to use"
    ),
):
    """Interactive config generator for agents and teams."""
    from llmling_textual.creator.creator_app import ConfigGeneratorApp

    app = ConfigGeneratorApp(model=model, provider=provider)  # type: ignore
    app.run()
