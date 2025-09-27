"""UI command for running configured UI."""

from __future__ import annotations

from typing import TYPE_CHECKING

import typer as t

from llmling_agent import AgentPool
from llmling_agent_cli import resolve_agent_config


if TYPE_CHECKING:
    from llmling_agent_ui.base import UIProvider


def ui_command(
    config_path: str = t.Option(None, "-c", "--config", help="Override config path"),
):
    """Start the UI configured in the manifest."""
    from llmling_agent_config.ui import (
        PromptToolkitUIConfig,
        StdlibUIConfig,
        TextualUIConfig,
    )

    try:
        config_path = resolve_agent_config(config_path)
        pool = AgentPool[None](config_path)
        ui_config = pool.manifest.ui
        match ui_config.type:
            case StdlibUIConfig():
                from llmling_agent_ui.stdlib_provider import StdlibUIProvider

                provider: UIProvider = StdlibUIProvider(ui_config)
            case PromptToolkitUIConfig():
                from llmling_agent_ui.prompt_toolkit_provider import (
                    PromptToolkitUIProvider,
                )

                provider = PromptToolkitUIProvider(ui_config)
            case TextualUIConfig():
                from llmling_agent_ui.textual_provider import TextualUIProvider

                provider = TextualUIProvider(ui_config)
            case _:
                msg = f"Unknown UI type: {ui_config.type}"
                raise ValueError(msg)  # noqa: TRY301

        provider.run_pool(pool)

    except Exception as e:
        t.echo(f"Error: {e}", err=True)
        raise t.Exit(1) from e
