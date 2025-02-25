"""Unified configuration for the complete vectorization pipeline."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from llmling_agent_config.embeddings import EmbeddingConfig, SentenceTransformersConfig
from llmling_agent_config.splitters import ChunkerConfig, MarkoChunkerConfig
from llmling_agent_config.vector_db import ChromaConfig, VectorStoreConfig


# Shorthand literals for common configurations
ChunkerShorthand = Literal[
    "markdown",  # MarkoChunkerConfig with defaults
    "langchain",  # LangChain recursive with defaults
    "llamaindex",  # LlamaIndex markdown with defaults
]

EmbeddingShorthand = Literal[
    "openai",  # OpenAI ada-002
    "bge-small",  # BGE small model
    "bge-large",  # BGE large model
    "minilm",  # all-MiniLM-L6-v2
]

VectorDBShorthand = Literal[
    "chroma",  # Local ChromaDB
    "chroma-memory",  # In-memory ChromaDB
    "qdrant",  # Local Qdrant
    "qdrant-memory",  # In-memory Qdrant
]


class VectorizationConfig(BaseModel):
    """Complete configuration for text vectorization pipeline."""

    chunker: ChunkerConfig | ChunkerShorthand = "markdown"
    """How to split text into chunks."""

    embedding: EmbeddingConfig | EmbeddingShorthand = "minilm"
    """How to generate embeddings."""

    store: VectorStoreConfig | VectorDBShorthand = "chroma-memory"
    """Where to store vectors."""

    model_config = ConfigDict(
        frozen=True,
        use_attribute_docstrings=True,
    )

    def resolve_chunker(self) -> ChunkerConfig:
        """Get full chunker config from shorthand or pass through existing."""
        match self.chunker:
            case "markdown":
                return MarkoChunkerConfig()
            case "langchain":
                from llmling_agent_config.splitters import LangChainChunkerConfig

                return LangChainChunkerConfig()
            case "llamaindex":
                from llmling_agent_config.splitters import LlamaIndexChunkerConfig

                return LlamaIndexChunkerConfig()
            case _:
                return self.chunker

    def resolve_embedding(self) -> EmbeddingConfig:
        """Get full embedding config from shorthand or pass through existing."""
        match self.embedding:
            case "openai":
                from llmling_agent_config.embeddings import OpenAIEmbeddingConfig

                return OpenAIEmbeddingConfig()
            case "bge-small":
                from llmling_agent_config.embeddings import BGEConfig

                return BGEConfig(model_name="BAAI/bge-small-en")
            case "bge-large":
                from llmling_agent_config.embeddings import BGEConfig

                return BGEConfig(model_name="BAAI/bge-large-en")
            case "minilm":
                return SentenceTransformersConfig()
            case _:
                return self.embedding

    def resolve_store(self) -> VectorStoreConfig:
        """Get full store config from shorthand or pass through existing."""
        match self.store:
            case "chroma":
                return ChromaConfig(persist_directory="./chroma_db")
            case "chroma-memory":
                return ChromaConfig()
            case "qdrant":
                from llmling_agent_config.vector_db import QdrantConfig

                return QdrantConfig(location="./qdrant_db")
            case "qdrant-memory":
                from llmling_agent_config.vector_db import QdrantConfig

                return QdrantConfig()
            case _:
                return self.store
