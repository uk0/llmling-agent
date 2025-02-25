"""Module providing different embedding model implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    import numpy as np


class EmbeddingProvider(ABC):
    """Base class for streaming embedding providers."""

    @abstractmethod
    def embed_stream(
        self,
        texts: AsyncIterator[str],
        batch_size: int = 8,
    ) -> AsyncIterator[np.ndarray]:
        """Stream embeddings one at a time."""
