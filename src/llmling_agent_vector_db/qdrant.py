"""RAG-based tool registry for semantic tool discovery."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast
import uuid

from llmling_agent.log import get_logger
from llmling_agent.vector_db import Metric, SearchResult, VectorStore


if TYPE_CHECKING:
    import numpy as np

    from llmling_agent_config.vector_db import QdrantConfig


logger = get_logger(__name__)


class QdrantVectorStore(VectorStore):
    """Qdrant implementation."""

    def __init__(self, config: QdrantConfig, vector_size: int = 1536):
        """Initialize Qdrant vector store.

        Args:
            config: Qdrant configuration
            vector_size: Size of vectors to store
        """
        import qdrant_client
        import qdrant_client.models

        # Create client based on configuration
        client_kwargs: dict[str, Any] = {}
        if config.url:
            client_kwargs["url"] = config.url
            if config.api_key:
                client_kwargs["api_key"] = config.api_key.get_secret_value()
        elif config.location:
            client_kwargs["location"] = config.location
        else:
            client_kwargs["location"] = ":memory:"

        client_kwargs["prefer_grpc"] = config.prefer_grpc

        self._client = qdrant_client.QdrantClient(**client_kwargs)
        self._collection_name = config.collection_name
        self._vector_size = vector_size

        # Check if collection exists
        collections = self._client.get_collections().collections
        collection_names = [c.name for c in collections]

        # Create collection if it doesn't exist
        if self._collection_name not in collection_names:
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=qdrant_client.models.VectorParams(
                    size=vector_size,
                    distance=qdrant_client.models.Distance.COSINE,
                ),
            )

        logger.info("Qdrant initialized with collection: %s", self._collection_name)

    def add(
        self,
        embedding: np.ndarray,
        metadata: dict[str, Any] | None = None,
        doc_id: str | None = None,
    ) -> str:
        """Add vector to Qdrant."""
        import qdrant_client.models

        # Generate ID if not provided
        if doc_id is None:
            doc_id = str(uuid.uuid4())

        # Convert numpy to list of floats (not string)
        vector_data = embedding.astype(float).tolist()
        vector_list = cast(list[float], vector_data)

        # Upsert vector
        self._client.upsert(
            collection_name=self._collection_name,
            points=[
                qdrant_client.models.PointStruct(
                    id=doc_id,
                    vector=vector_list,  # Now properly typed as list[float]
                    payload=metadata or {},
                )
            ],
        )

        return doc_id

    def search(
        self,
        query_vector: np.ndarray,
        limit: int = 5,
        filters: dict[str, Any] | None = None,
        metric: Metric = "cosine",
        search_params: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search Qdrant for similar vectors."""
        from qdrant_client import models

        # Convert numpy to list of floats
        vector_data = query_vector.astype(float).tolist()
        vector_list = cast(list[float], vector_data)

        # Build filter if needed
        filter_query = None
        if filters:
            conditions = []
            for field_name, value in filters.items():
                if isinstance(value, list):
                    conditions.append(
                        models.FieldCondition(
                            key=field_name,
                            match=models.MatchAny(any=value),
                        )
                    )
                else:
                    conditions.append(
                        models.FieldCondition(
                            key=field_name,
                            match=models.MatchValue(value=value),
                        )
                    )

            if conditions:
                filter_query = models.Filter(must=conditions)

        # Execute search
        results = self._client.search(
            collection_name=self._collection_name,
            query_vector=vector_list,
            limit=limit,
            with_payload=True,
            filter=filter_query,
        )

        # Format results with proper type conversion
        return [
            SearchResult(str(hit.id), hit.score, dict(hit.payload or {}))
            for hit in results
        ]

    def delete(self, doc_id: str) -> bool:
        """Delete vector by ID."""
        from qdrant_client import models
        from qdrant_client.http.exceptions import UnexpectedResponse

        selector = models.PointIdsList(points=[doc_id])
        try:
            self._client.delete(
                collection_name=self._collection_name,
                points_selector=selector,
            )
        except UnexpectedResponse:
            return False
        else:
            return True
