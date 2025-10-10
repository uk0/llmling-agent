"""System prompts configuration for agents."""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ImportString


class StaticPromptConfig(BaseModel):
    """Configuration for a static text prompt."""

    type: Literal["static"] = Field("static", init=False)
    content: str
    """The prompt text content."""

    model_config = ConfigDict(frozen=True)


class FilePromptConfig(BaseModel):
    """Configuration for a file-based Jinja template prompt."""

    type: Literal["file"] = Field("file", init=False)
    path: str
    """Path to the Jinja template file."""

    variables: dict[str, Any] = Field(default_factory=dict)
    """Variables to pass to the template."""

    model_config = ConfigDict(frozen=True)


class LibraryPromptConfig(BaseModel):
    """Configuration for a library reference prompt."""

    type: Literal["library"] = Field("library", init=False)
    reference: str
    """Library prompt reference identifier."""

    model_config = ConfigDict(frozen=True)


class FunctionPromptConfig(BaseModel):
    """Configuration for a function-generated prompt."""

    type: Literal["function"] = Field("function", init=False)
    function: ImportString[Callable[..., str]]
    """Import path to the function that generates the prompt."""

    arguments: dict[str, Any] = Field(default_factory=dict)
    """Arguments to pass to the function."""

    model_config = ConfigDict(frozen=True)


PromptConfig = Annotated[
    StaticPromptConfig | FilePromptConfig | LibraryPromptConfig | FunctionPromptConfig,
    Field(discriminator="type"),
]
"""Union type for different prompt configuration types."""


__all__ = [
    "FilePromptConfig",
    "FunctionPromptConfig",
    "LibraryPromptConfig",
    "PromptConfig",
    "StaticPromptConfig",
]
