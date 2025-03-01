"""LlamaIndex-based text chunker implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from llmling_agent.log import get_logger
from llmling_agent.splitters import TextChunk, TextChunker


if TYPE_CHECKING:
    from llmling_agent_config.splitters import LlamaIndexChunkerConfig


logger = get_logger(__name__)


class LlamaIndexChunker(TextChunker):
    """Text chunker using LlamaIndex text splitters."""

    def __init__(self, config: LlamaIndexChunkerConfig):
        """Initialize with configuration."""
        self.config = config

    def split(
        self, text: str, document_metadata: dict[str, Any] | None = None
    ) -> list[TextChunk]:
        """Split text using LlamaIndex splitter."""
        from llama_index.core import Document
        from llama_index.core.node_parser import (
            MarkdownNodeParser,
            SentenceWindowNodeParser,
            SimpleNodeParser,
            TokenTextSplitter,
        )

        doc_meta = document_metadata or {}

        # Create LlamaIndex document with metadata
        document = Document(text=text, metadata=doc_meta)

        # Configure the appropriate node parser
        match self.config.chunker_type:
            case "markdown":
                parser = MarkdownNodeParser.from_defaults(
                    include_metadata=self.config.include_metadata,
                    include_prev_next_rel=self.config.include_prev_next_rel,
                )
            case "sentence":
                parser = SentenceWindowNodeParser.from_defaults(
                    window_size=self.config.chunk_size,
                    include_metadata=self.config.include_metadata,
                    include_prev_next_rel=self.config.include_prev_next_rel,
                )
            case "token":
                text_splitter = TokenTextSplitter(
                    chunk_size=self.config.chunk_size,
                    chunk_overlap=self.config.chunk_overlap,
                )
                parser = SimpleNodeParser.from_defaults(
                    text_splitter=text_splitter,
                    include_metadata=self.config.include_metadata,
                    include_prev_next_rel=self.config.include_prev_next_rel,
                )
            case "fixed":
                from llama_index.core.node_parser.text import SplitterType

                parser = SimpleNodeParser.from_defaults(
                    chunk_size=self.config.chunk_size,
                    chunk_overlap=self.config.chunk_overlap,
                    splitter_type=SplitterType.CHARACTER,
                    include_metadata=self.config.include_metadata,
                    include_prev_next_rel=self.config.include_prev_next_rel,
                )
            case _:
                logger.warning(
                    "Unknown chunker type %r, falling back to markdown",
                    self.config.chunker_type,
                )
                parser = MarkdownNodeParser.from_defaults(
                    include_metadata=self.config.include_metadata,
                    include_prev_next_rel=self.config.include_prev_next_rel,
                )

        # Parse the document into nodes
        nodes = parser.get_nodes_from_documents([document])

        # Convert LlamaIndex nodes to our TextChunk format
        result = []
        for i, node in enumerate(nodes):
            # Get text and metadata
            node_text = node.get_content()

            # Combine document metadata with node metadata
            node_meta = {
                **doc_meta,
                "chunk_index": i,
                "llamaindex_chunker": self.config.chunker_type,
            }

            # Add node metadata if available
            if node.metadata:
                for k, v in node.metadata.items():
                    if k not in node_meta and isinstance(v, str | int | float | bool):
                        node_meta[k] = v

            result.append(TextChunk(node_text, node_meta))

        return result
