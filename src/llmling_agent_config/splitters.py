"""Configuration models for text chunking."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BaseChunkerConfig(BaseModel):
    """Base configuration for text chunkers."""

    type: str = Field(init=False)
    """Type identifier for the chunker."""

    chunk_overlap: int = 200
    """Number of characters to overlap between chunks."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)


class LangChainChunkerConfig(BaseChunkerConfig):
    """Configuration for LangChain chunkers."""

    type: Literal["langchain"] = Field(default="langchain", init=False)

    chunker_type: Literal["recursive", "markdown", "character"] = "recursive"
    """Which LangChain chunker to use."""

    chunk_size: int = 1000
    """Target size of chunks."""


class MarkoChunkerConfig(BaseChunkerConfig):
    """Configuration for marko-based markdown chunker."""

    type: Literal["marko"] = Field(default="marko", init=False)

    split_on: Literal["headers", "paragraphs", "blocks"] = "headers"
    """How to split the markdown."""

    min_header_level: int = Field(default=2, ge=1, le=6)
    """Minimum header level to split on (if splitting on headers)."""

    combine_small_sections: bool = True
    """Whether to combine small sections with neighbors."""

    min_section_length: int = Field(default=100, ge=0)
    """Minimum length for a section before combining."""

    @model_validator(mode="after")
    def validate_section_length(self) -> MarkoChunkerConfig:
        """Ensure min_section_length is only used with combine_small_sections."""
        if not self.combine_small_sections and self.min_section_length > 0:
            msg = "min_section_length only valid when combine_small_sections=True"
            raise ValueError(msg)
        return self


class LlamaIndexChunkerConfig(BaseChunkerConfig):
    """Configuration for LlamaIndex chunkers."""

    type: Literal["llamaindex"] = Field(default="llamaindex", init=False)

    chunker_type: Literal["sentence", "token", "fixed", "markdown"] = "markdown"
    """Which LlamaIndex chunker to use."""

    chunk_size: int = 1000
    """Target size of chunks."""

    include_metadata: bool = True
    """Whether to include document metadata in chunks."""

    include_prev_next_rel: bool = False
    """Whether to track relationships between chunks."""


# Union type for all chunker configs
ChunkerConfig = Annotated[
    LangChainChunkerConfig | LlamaIndexChunkerConfig | MarkoChunkerConfig,
    Field(discriminator="type"),
]
