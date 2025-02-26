"""Vector store implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from llmling_agent.log import get_logger


if TYPE_CHECKING:
    import numpy as np


logger = get_logger(__name__)


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
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """Search for similar vectors.

        Args:
            query_vector: Query embedding
            limit: Maximum number of results
            filters: Optional metadata filters

        Returns:
            List of (doc_id, similarity_score, metadata) tuples
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
