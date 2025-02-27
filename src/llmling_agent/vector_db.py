"""Vector store implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from llmling_agent.log import get_logger


if TYPE_CHECKING:
    import numpy as np


logger = get_logger(__name__)

Metric = Literal["cosine", "euclidean", "dot"]


@dataclass(frozen=True)
class SearchResult:
    """A single vector search result."""

    doc_id: str
    score: float  # similarity score between 0-1
    metadata: dict[str, str | int | float | bool | list[str]]


class VectorStore:
    """Base class for vector stores."""

    def add(
        self,
        embedding: np.ndarray,
        metadata: dict[str, Any] | None = None,
        doc_id: str | None = None,
    ) -> str:
        """Add vector to the store.

        Args:
            embedding: Vector embedding
            metadata: Optional metadata to store with the vector
            doc_id: Optional document ID (auto-generated if None)

        Returns:
            The document ID (either provided or auto-generated)
        """
        raise NotImplementedError

    def search(
        self,
        query_vector: np.ndarray,
        limit: int = 5,
        filters: dict[str, Any] | None = None,
        metric: Metric = "cosine",
        search_params: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for similar vectors.

        Added:
        - metric: Distance metric to use
        - search_params: Provider-specific search parameters (e.g. ef_search for HNSW)
        """
        raise NotImplementedError

    def delete(self, doc_id: str) -> bool:
        """Delete vector by ID.

        Args:
            doc_id: Document identifier

        Returns:
            True if deleted, False if not found
        """
        raise NotImplementedError

    def add_batch(
        self,
        embeddings: list[np.ndarray],
        metadatas: list[dict[str, Any]] | None = None,
        doc_ids: list[str] | None = None,
    ) -> list[str]:
        """Add multiple vectors to the store.

        Default implementation falls back to single adds.
        """
        meta = [None] * len(embeddings) if metadatas is None else metadatas
        doc_ids_ = [None] * len(embeddings) if doc_ids is None else doc_ids

        return [
            self.add(emb, meta, id_)
            for emb, meta, id_ in zip(embeddings, meta, doc_ids_, strict=False)
        ]

    def search_batch(
        self,
        query_vectors: list[np.ndarray],
        limit: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[list[SearchResult]]:
        """Search for multiple query vectors.

        Default implementation falls back to single searches.
        """
        return [self.search(qv, limit, filters) for qv in query_vectors]
