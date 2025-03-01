"""Unified configuration for the complete RAG pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field


if TYPE_CHECKING:
    from llmling_agent_config.converters import ConverterConfig
    from llmling_agent_config.embeddings import EmbeddingConfig
    from llmling_agent_config.splitters import ChunkerConfig
    from llmling_agent_config.vector_db import VectorStoreConfig

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

ConverterShorthand = Literal[
    "docling",  # Docling with defaults
    "markitdown",  # MarkItDown with defaults
    "plain",  # Plain text with defaults
]


class RAGPipelineConfig(BaseModel):
    """Complete configuration for text vectorization pipeline."""

    paths: list[str]
    """Input paths to process (files/folders/URLs)"""

    converter: ConverterConfig | ConverterShorthand = "docling"
    """How to convert documents to text."""

    chunker: ChunkerConfig | ChunkerShorthand = "markdown"
    """How to split text into chunks."""

    embedding: EmbeddingConfig | EmbeddingShorthand = "minilm"
    """How to generate embeddings."""

    store: VectorStoreConfig | VectorDBShorthand = "chroma-memory"
    """Where to store vectors."""

    batch_size: int = Field(default=8, gt=0)
    """Batch size for embeddings."""

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)

    def resolve_converter(self) -> ConverterConfig:
        """Get full converter config from shorthand or pass through existing."""
        match self.converter:
            case "docling":
                from llmling_agent_config.converters import DoclingConverterConfig

                return DoclingConverterConfig()
            case "markitdown":
                from llmling_agent_config.converters import MarkItDownConfig

                return MarkItDownConfig()
            case "plain":
                from llmling_agent_config.converters import PlainConverterConfig

                return PlainConverterConfig()
            case _:
                return self.converter

    def resolve_chunker(self) -> ChunkerConfig:
        """Get full chunker config from shorthand or pass through existing."""
        match self.chunker:
            case "markdown":
                from llmling_agent_config.splitters import MarkoChunkerConfig

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
                from llmling_agent_config.embeddings import SentenceTransformersConfig

                return SentenceTransformersConfig()
            case _:
                return self.embedding

    def resolve_store(self) -> VectorStoreConfig:
        """Get full store config from shorthand or pass through existing."""
        match self.store:
            case "chroma":
                from llmling_agent_config.vector_db import ChromaConfig

                return ChromaConfig(persist_directory="./chroma_db")
            case "chroma-memory":
                from llmling_agent_config.vector_db import ChromaConfig

                return ChromaConfig()
            case "qdrant":
                from llmling_agent_config.vector_db import QdrantConfig

                return QdrantConfig(location="./qdrant_db")
            case "qdrant-memory":
                from llmling_agent_config.vector_db import QdrantConfig

                return QdrantConfig()
            case _:
                return self.store
