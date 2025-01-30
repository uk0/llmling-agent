from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class BaseConverterConfig(BaseModel):
    """Base configuration for document converters."""

    type: str = Field(init=False)
    """Type discriminator for converter configs."""

    enabled: bool = True
    """Whether this converter is currently active."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True, extra="forbid")


class DoclingConverterConfig(BaseConverterConfig):
    """Configuration for docling-based converter."""

    type: Literal["docling"] = Field("docling", init=False)
    """Type discriminator for docling converter."""

    max_size: int | None = None
    """Optional size limit in bytes."""


class MarkItDownConfig(BaseConverterConfig):
    """Configuration for MarkItDown-based converter."""

    type: Literal["markitdown"] = Field("markitdown", init=False)
    """Type discriminator for MarkItDown converter."""

    max_size: int | None = None
    """Optional size limit in bytes."""


class PlainConverterConfig(BaseConverterConfig):
    """Configuration for plain text fallback converter."""

    type: Literal["plain"] = Field("plain", init=False)
    """Type discriminator for plain text converter."""

    force: bool = False
    """Whether to attempt converting any file type."""


ConverterConfig = Annotated[
    DoclingConverterConfig | MarkItDownConfig | PlainConverterConfig,
    Field(discriminator="type"),
]


class ConversionConfig(BaseModel):
    """Global conversion configuration."""

    providers: list[ConverterConfig] | None = None
    """List of configured converter providers."""

    default_provider: str | None = None
    """Name of default provider for conversions."""

    max_size: int | None = None
    """Global size limit for all converters."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True, extra="forbid")
