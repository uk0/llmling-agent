"""Embedding model configuration."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr


class BaseEmbeddingConfig(BaseModel):
    """Base configuration for embedding models."""

    type: str = Field(init=False)
    """Type identifier for the embedding model."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)


class SentenceTransformersConfig(BaseEmbeddingConfig):
    """Configuration for sentence-transformers models."""

    type: Literal["sentence-transformers"] = Field(
        default="sentence-transformers",
        init=False,
    )

    model_name: str = "all-MiniLM-L6-v2"
    """Name of the model to use."""

    use_gpu: bool = False
    """Whether to use GPU for inference."""

    batch_size: int = 32
    """Batch size for inference."""


class OpenAIEmbeddingConfig(BaseEmbeddingConfig):
    """Configuration for OpenAI's embedding API."""

    type: Literal["openai"] = Field(default="openai", init=False)

    model: str = "text-embedding-ada-002"
    """Model to use."""

    api_key: SecretStr | None = None
    """OpenAI API key."""


class BGEConfig(BaseEmbeddingConfig):
    """Configuration for BGE embedding models."""

    type: Literal["bge"] = Field(default="bge", init=False)

    model_name: str = "BAAI/bge-small-en"
    """Name/size of BGE model to use."""

    use_gpu: bool = False
    """Whether to use GPU for inference."""

    batch_size: int = 32
    """Batch size for inference."""


# Union type for embedding configs
EmbeddingConfig = Annotated[
    SentenceTransformersConfig | OpenAIEmbeddingConfig | BGEConfig,
    Field(discriminator="type"),
]
