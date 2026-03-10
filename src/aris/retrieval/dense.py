"""Dense embedding-based retrieval from local index."""

from __future__ import annotations

import logging

from aris.core.models import ScoredDocument
from aris.index.dense_store import DenseStore

logger = logging.getLogger(__name__)


class DenseRetriever:
    """Retrieves documents using embedding similarity from the dense index."""

    def __init__(self, store: DenseStore) -> None:
        self._store = store

    async def retrieve(self, query: str, num_results: int = 100) -> list[ScoredDocument]:
        """Retrieve documents by embedding similarity."""
        if self._store.count == 0:
            return []

        results = self._store.query(query, n_results=num_results)
        scored = []
        for doc, score in results:
            scored.append(ScoredDocument(
                document=doc,
                retrieval_score=score,
                final_score=score,
            ))
        return scored
