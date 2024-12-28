"""Models for sources configuration."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


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
