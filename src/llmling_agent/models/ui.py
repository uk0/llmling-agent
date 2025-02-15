"""UI Provider Configuration."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class TriggerConfig(BaseModel):
    """Configuration for initial trigger on UI startup."""

    node_name: str
    """Name of the node (agent/team) to trigger."""

    prompts: list[str]
    """Prompts to send."""

    exit_after: bool = False
    """Whether to exit after trigger completes."""


class BaseUIConfig(BaseModel):
    """Base configuration for UI providers."""

    type: str = Field(init=False)
    """Type discriminator for UI configs."""

    trigger: TriggerConfig | None = None
    """Optional trigger configuration to run on startup."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True, extra="forbid")


class StdlibUIConfig(BaseUIConfig):
    """Configuration for basic CLI interface."""

    type: Literal["cli"] = Field("cli", init=False)
    """Basic CLI interface."""

    show_messages: bool = True
    """Show all messages or just final responses."""

    detail_level: Literal["simple", "detailed", "markdown"] = "simple"
    """Output detail level."""

    show_metadata: bool = False
    """Show message metadata."""

    show_costs: bool = False
    """Show token usage and costs."""


class GradioUIConfig(BaseUIConfig):
    """Configuration for web interface."""

    type: Literal["web"] = Field("web", init=False)
    """Web interface."""

    host: str = "127.0.0.1"
    """Host to bind to."""

    port: int | None = None
    """Port to bind to."""

    share: bool = False
    """Create public URL."""

    theme: Literal["soft", "base", "monochrome", "glass"] = "soft"
    """UI theme."""


class PromptToolkitUIConfig(BaseUIConfig):
    """Configuration for prompt-toolkit interface."""

    type: Literal["prompt"] = Field("prompt", init=False)
    """Prompt-toolkit interface."""

    stream: bool = True
    """Enable response streaming."""


class TextualUIConfig(BaseUIConfig):
    """Configuration for Textual TUI interface."""

    type: Literal["textual"] = Field("textual", init=False)
    """Textual terminal UI."""


UIConfig = Annotated[
    StdlibUIConfig | GradioUIConfig | PromptToolkitUIConfig | TextualUIConfig,
    Field(discriminator="type"),
]
