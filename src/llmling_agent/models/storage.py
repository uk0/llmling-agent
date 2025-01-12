from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


LogFormat = Literal["chronological", "conversations"]


class BaseStorageProviderConfig(BaseModel):
    type: str = Field(init=False)

    enabled: bool = True
    """Whether this provider is active"""

    log_messages: bool = True
    """Whether to log messages"""

    log_conversations: bool = True
    """Whether to log conversations"""

    log_tool_calls: bool = True
    """Whether to log tool calls"""

    log_commands: bool = True
    """Whether to log command executions"""
    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)


class SQLStorageConfig(BaseStorageProviderConfig):
    """SQL database storage configuration."""

    type: Literal["sql"] = Field("sql", init=False)
    url: str
    """Database URL (e.g. sqlite:///history.db)"""
    pool_size: int = 5
    """Connection pool size"""


class TextLogConfig(BaseStorageProviderConfig):
    """Text log configuration."""

    type: Literal["text_file"] = Field("text_file", init=False)
    path: str
    """Path to log file"""
    format: LogFormat = "chronological"
    """Log format template to use"""
    template: Literal["chronological", "conversations"] | str | None = "chronological"  # noqa: PYI051
    """Template to use: either predefined name or path to custom template"""
    encoding: str = "utf-8"
    """File encoding"""


# Config:
class FileStorageConfig(BaseStorageProviderConfig):
    """File storage configuration."""

    type: Literal["file"] = Field("file", init=False)
    path: str
    """Path to storage file (extension determines format unless specified)"""
    format: Literal["auto", "yaml", "json", "toml"] = "auto"
    """Storage format (auto=detect from extension)"""


StorageProviderConfig = Annotated[
    SQLStorageConfig | FileStorageConfig | TextLogConfig, Field(discriminator="type")
]


class StorageConfig(BaseModel):
    """Global storage configuration."""

    providers: list[StorageProviderConfig] = Field(default_factory=list)
    """List of configured storage providers"""

    default_provider: str | None = None
    """Name of default provider for history queries.
    If None, uses first configured provider."""

    log_messages: bool = True
    """Whether to log messages."""

    log_conversations: bool = True
    """Whether to log conversations."""

    log_tool_calls: bool = True
    """Whether to log tool calls."""

    log_commands: bool = True
    """Whether to log command executions."""

    model_config = ConfigDict(frozen=True)
