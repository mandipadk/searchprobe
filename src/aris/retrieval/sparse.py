"""Sparse BM25 retrieval from local index."""

from __future__ import annotations

import logging

from aris.core.models import ScoredDocument
from aris.index.sparse_store import SparseStore

logger = logging.getLogger(__name__)


class SparseRetriever:
    """Retrieves documents using BM25 keyword matching."""

    def __init__(self, store: SparseStore) -> None:
        self._store = store

    async def retrieve(self, query: str, num_results: int = 100) -> list[ScoredDocument]:
        if self._store.count == 0:
            return []

        results = self._store.query(query, n_results=num_results)
        return [
            ScoredDocument(document=doc, retrieval_score=score, final_score=score)
            for doc, score in results
        ]
