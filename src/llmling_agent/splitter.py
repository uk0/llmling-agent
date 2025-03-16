"""Base classes for text chunking implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import AsyncIterable, Iterable


class TextChunk:
    """Chunk of text with metadata."""

    def __init__(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
    ):
        self.text = text
        self.metadata = metadata or {}


class TextChunker(ABC):
    """Base class for text chunkers."""

    @abstractmethod
    def split(
        self, text: str, document_metadata: dict[str, Any] | None = None
    ) -> list[TextChunk]:
        """Split text into chunks."""
        raise NotImplementedError

    async def split_async(
        self, text: str, document_metadata: dict[str, Any] | None = None
    ) -> AsyncIterable[TextChunk]:
        """Split text asynchronously into chunks."""
        # Default implementation, override for true async processing
        for chunk in self.split(text, document_metadata):
            yield chunk

    def split_texts(
        self, texts: Iterable[str | tuple[str, dict[str, Any]]]
    ) -> list[TextChunk]:
        """Split multiple texts into chunks."""
        result: list[TextChunk] = []

        for item in texts:
            if isinstance(item, tuple):
                text, metadata = item
            else:
                text, metadata = item, None

            result.extend(self.split(text, metadata))

        return result
