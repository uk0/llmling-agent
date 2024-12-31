"""Models for sources configuration."""

from __future__ import annotations

from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, Field, model_validator


class ContextSourceConfig(BaseModel):
    """Configuration for a context source."""

    type: str = Field(init=False)
    """Type discriminator for context sources"""

    convert_to_md: bool = False
    """Whether to convert content to markdown"""

    metadata: dict[str, str] = Field(default_factory=dict)
    """Additional metadata to include with the message"""


class FileContextSource(ContextSourceConfig):
    """Load context from a file."""

    type: Literal["file"] = Field("file", init=False)
    path: str
    """Path to file (supports glob patterns)"""


class ResourceContextSource(ContextSourceConfig):
    """Load context from a resource."""

    type: Literal["resource"] = Field("resource", init=False)
    name: str
    """Name of the resource"""
    arguments: dict[str, Any] = Field(default_factory=dict)
    """Optional arguments for resource loading"""


class PromptContextSource(ContextSourceConfig):
    """Load context from a prompt."""

    type: Literal["prompt"] = Field("prompt", init=False)
    name: str
    """Name of the prompt"""
    arguments: dict[str, Any] = Field(default_factory=dict)
    """Arguments for prompt rendering"""


ContextSource = Annotated[
    FileContextSource | ResourceContextSource | PromptContextSource,
    Field(discriminator="type"),
]


class Knowledge(BaseModel):
    """Collection of context sources for an agent."""

    paths: list[str | FileContextSource] = Field(default_factory=list)
    """Files or paths to load as context."""

    resources: list[str | ResourceContextSource] = Field(default_factory=list)
    """Resources to load as context."""

    prompts: list[str | PromptContextSource] = Field(default_factory=list)
    """Prompts to load as context."""

    convert_paths_to_markdown: bool = False
    """Whether to convert file contents to markdown by default."""

    @model_validator(mode="after")
    def convert_simple_configs(self) -> Self:
        """Convert simple string inputs to full source configs."""
        self.paths = [
            (
                FileContextSource(path=p, convert_to_md=self.convert_paths_to_markdown)
                if isinstance(p, str)
                else p
            )
            for p in self.paths
        ]
        self.resources = [
            ResourceContextSource(name=r) if isinstance(r, str) else r
            for r in self.resources
        ]
        self.prompts = [
            PromptContextSource(name=p) if isinstance(p, str) else p for p in self.prompts
        ]
        return self
