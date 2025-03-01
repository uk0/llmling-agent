"""Markdown chunking implementation using the marko library."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from marko import Renderer
from marko.md_renderer import MarkdownRenderer

from llmling_agent.log import get_logger
from llmling_agent.splitters import TextChunk, TextChunker


if TYPE_CHECKING:
    from marko.block import Document, Heading, Paragraph

    from llmling_agent_config.splitters import MarkoChunkerConfig

logger = get_logger(__name__)


class SectionExtractor(Renderer):
    """Renderer that extracts sections from markdown."""

    def __init__(self, min_header_level: int = 2):
        """Initialize with minimum header level for splitting.

        Args:
            min_header_level: Minimum header level to split on (1-6)
        """
        self.min_header_level = min_header_level
        self.sections: list[tuple[str | None, str]] = []
        self.current_title: str | None = None
        self.current_content: list[str] = []
        self._rendered_children = ""

    def render_heading(self, element: Heading) -> str:
        """Handle a heading element."""
        title = self.render_children(element).strip()

        # If we hit a heading at our target level or above, start new section
        if element.level <= self.min_header_level:
            # Save current section if it exists
            if self.current_content:
                content = "\n\n".join(self.current_content)
                self.sections.append((self.current_title, content))
                self.current_content = []

            self.current_title = title

        return f"{'#' * element.level} {title}\n\n"

    def render_document(self, element: Document) -> str:
        """Process the entire document."""
        result = self.render_children(element)

        # Don't forget the last section
        if self.current_content:
            content = "\n\n".join(self.current_content)
            self.sections.append((self.current_title, content))

        return result

    def render_paragraph(self, element: Paragraph) -> str:
        """Handle a paragraph element."""
        content = self.render_children(element).strip()
        self.current_content.append(content)
        return f"{content}\n\n"


class MarkoChunker(TextChunker):
    """Marko-based markdown chunker."""

    def __init__(self, config: MarkoChunkerConfig):
        """Initialize with configuration.

        Args:
            config: Chunker configuration
        """
        self.min_header_level = config.min_header_level
        self.split_on = config.split_on
        self.combine_small_sections = config.combine_small_sections
        self.min_section_length = config.min_section_length
        self.chunk_overlap = config.chunk_overlap

        # For rebuilding markdown sections
        self.md_renderer = MarkdownRenderer()

    def split(
        self, text: str, document_metadata: dict[str, Any] | None = None
    ) -> list[TextChunk]:
        """Split text into chunks based on markdown structure."""
        doc_meta = document_metadata or {}

        match self.split_on:
            case "headers":
                return self._split_by_headers(text, doc_meta)
            case "paragraphs":
                return self._split_by_paragraphs(text, doc_meta)
            case "blocks":
                return self._split_by_blocks(text, doc_meta)
            case _:
                logger.warning(
                    "Unknown split method %r, falling back to headers", self.split_on
                )
                return self._split_by_headers(text, doc_meta)

    def _split_by_headers(self, text: str, doc_meta: dict[str, Any]) -> list[TextChunk]:
        """Split text by headers at or above specified level."""
        import marko

        # Parse the markdown
        parsed = marko.parse(text)

        # Extract sections
        extractor = SectionExtractor(self.min_header_level)
        extractor.render(parsed)

        # Create chunks from sections
        chunks: list[TextChunk] = []
        for i, (title, content) in enumerate(extractor.sections):
            # Skip empty sections
            if not content.strip():
                continue

            # Combine title with content
            chunk_text = f"# {title}\n\n{content}" if title else content

            # Make metadata
            chunk_meta = {
                **doc_meta,
                "chunk_index": i,
                "section_title": title or "",
            }

            chunks.append(TextChunk(chunk_text, chunk_meta))

        # Post-process: combine small chunks if needed
        if self.combine_small_sections and len(chunks) > 1:
            return self._combine_small_chunks(chunks)

        return chunks

    def _split_by_paragraphs(
        self, text: str, doc_meta: dict[str, Any]
    ) -> list[TextChunk]:
        """Split text by paragraphs (blank lines)."""
        paragraphs = [p.strip() for p in text.split("\n\n")]
        paragraphs = [p for p in paragraphs if p]  # Remove empty paragraphs

        chunks = []
        for i, paragraph in enumerate(paragraphs):
            chunk_meta = {
                **doc_meta,
                "chunk_index": i,
            }
            chunks.append(TextChunk(paragraph, chunk_meta))

        # Post-process: combine small chunks if needed
        if self.combine_small_sections and len(chunks) > 1:
            return self._combine_small_chunks(chunks)

        return chunks

    def _split_by_blocks(self, text: str, doc_meta: dict[str, Any]) -> list[TextChunk]:
        """Split text by markdown blocks."""
        import marko
        from marko.element import BlockElement

        # Parse the markdown
        parsed = marko.parse(text)

        # Extract top-level blocks
        blocks = []
        for child in parsed.children:
            if isinstance(child, BlockElement):
                block_text = self.md_renderer.render(child)
                if block_text.strip():
                    blocks.append(block_text)

        # Create chunks
        chunks = []
        for i, block in enumerate(blocks):
            chunk_meta = {
                **doc_meta,
                "chunk_index": i,
            }
            chunks.append(TextChunk(block, chunk_meta))

        # Post-process: combine small chunks if needed
        if self.combine_small_sections and len(chunks) > 1:
            return self._combine_small_chunks(chunks)

        return chunks

    def _combine_small_chunks(self, chunks: list[TextChunk]) -> list[TextChunk]:
        """Combine chunks that are smaller than min_section_length."""
        if not chunks:
            return chunks

        result: list[TextChunk] = []
        current: list[TextChunk] = []
        current_length = 0

        for chunk in chunks:
            chunk_len = len(chunk.text)

            # If adding this chunk exceeds our target size, finalize the current chunk
            if current and current_length + chunk_len > self.min_section_length:
                combined_text = "\n\n".join(c.text for c in current)
                combined_meta = {
                    **current[0].metadata,
                    "combined_chunks": len(current),
                }
                result.append(TextChunk(combined_text, combined_meta))
                current = []
                current_length = 0

            # Add this chunk to current collection
            current.append(chunk)
            current_length += chunk_len

        # Don't forget the last batch
        if current:
            combined_text = "\n\n".join(c.text for c in current)
            combined_meta = {
                **current[0].metadata,
                "combined_chunks": len(current),
            }
            result.append(TextChunk(combined_text, combined_meta))

        return result
