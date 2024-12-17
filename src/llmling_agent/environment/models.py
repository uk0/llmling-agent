from __future__ import annotations

from typing import Annotated, Literal

from llmling import Config  # noqa: TC002
from pydantic import BaseModel, ConfigDict, Field
from upath import UPath


class BaseEnvironment(BaseModel):
    """Base class for environment configurations."""

    type: str = Field(init=False)
    """Type discriminator for Environments."""

    config_file_path: str | None = None
    """Path to agent config file for resolving relative paths"""

    model_config = ConfigDict(frozen=True)

    def get_display_name(self) -> str:
        """Get human-readable environment identifier."""
        raise NotImplementedError

    def get_file_path(self) -> str | None:
        """Get file path if available."""
        return None


class FileEnvironment(BaseEnvironment):
    """File-based environment configuration.

    Loads environment settings from external YAML files, supporting:
    - Reusable environment configurations
    - Separation of concerns
    - Environment sharing between agents
    - Version control of environment settings
    """

    type: Literal["file"] = Field(default="file", init=False)
    uri: str = Field(description="Path to environment file", min_length=1)

    def get_display_name(self) -> str:
        return f"File: {self.uri}"

    def get_file_path(self) -> str:
        if self.config_file_path:
            base_dir = UPath(self.config_file_path).parent
            return str(base_dir / self.uri)
        return self.uri


class InlineEnvironment(BaseEnvironment):
    """Direct environment configuration without external files.

    Allows embedding complete environment settings directly in the agent
    configuration instead of referencing external files. Useful for:
    - Self-contained configurations
    - Testing and development
    - Simple agent setups
    """

    type: Literal["inline"] = Field(default="inline", init=False)
    uri: str | None = None
    config: Config = Field(..., description="Inline configuration")

    def get_display_name(self) -> str:
        return f"Inline: {self.uri}" if self.uri else "Inline configuration"


AgentEnvironment = Annotated[
    FileEnvironment | InlineEnvironment, Field(discriminator="type")
]
