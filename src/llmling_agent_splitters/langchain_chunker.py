"""LangChain-based text chunker implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from llmling_agent.log import get_logger
from llmling_agent.splitter import TextChunk, TextChunker


if TYPE_CHECKING:
    from langchain.text_splitter import TextSplitter

    from llmling_agent_config.splitters import LangChainChunkerConfig


logger = get_logger(__name__)


class LangChainChunker(TextChunker):
    """Text chunker using LangChain's text splitters."""

    def __init__(self, config: LangChainChunkerConfig):
        """Initialize with configuration."""
        self.config = config
        self._splitter = self._create_splitter()

    def _create_splitter(self) -> TextSplitter:
        """Create the appropriate LangChain splitter."""
        chunk_size = self.config.chunk_size
        chunk_overlap = self.config.chunk_overlap

        match self.config.chunker_type:
            case "recursive":
                from langchain.text_splitter import RecursiveCharacterTextSplitter

                return RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            case "markdown":
                from langchain.text_splitter import MarkdownTextSplitter

                return MarkdownTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            case "character":
                from langchain.text_splitter import CharacterTextSplitter

                return CharacterTextSplitter(
                    chunk_size=chunk_size, chunk_overlap=chunk_overlap
                )
            case _:
                logger.warning(
                    "Unknown chunker type %r, falling back to recursive",
                    self.config.chunker_type,
                )
                from langchain.text_splitter import RecursiveCharacterTextSplitter

                return RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )

    def split(
        self, text: str, document_metadata: dict[str, Any] | None = None
    ) -> list[TextChunk]:
        """Split text using LangChain splitter."""
        doc_meta = document_metadata or {}

        # Split the text
        chunks = self._splitter.split_text(text)

        # Convert to TextChunk objects with metadata
        result = []
        for i, chunk in enumerate(chunks):
            chunk_meta = {
                **doc_meta,
                "chunk_index": i,
                "langchain_chunker": self.config.chunker_type,
            }
            result.append(TextChunk(chunk, chunk_meta))

        return result
