"""BM25 sparse index using rank-bm25."""

from __future__ import annotations

import logging
import re

from aris.core.models import Document

logger = logging.getLogger(__name__)


class SparseStore:
    """BM25-based sparse retrieval index."""

    def __init__(self) -> None:
        self._documents: list[Document] = []
        self._tokenized_corpus: list[list[str]] = []
        self._bm25 = None

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    def add_documents(self, documents: list[Document]) -> None:
        """Add documents to the BM25 index."""
        from rank_bm25 import BM25Okapi

        for doc in documents:
            text = f"{doc.title} {doc.content or doc.snippet}"
            if text.strip():
                self._documents.append(doc)
                self._tokenized_corpus.append(self._tokenize(text))

        if self._tokenized_corpus:
            self._bm25 = BM25Okapi(self._tokenized_corpus)

    def query(self, query_text: str, n_results: int = 100) -> list[tuple[Document, float]]:
        """Query the BM25 index. Returns (Document, score) pairs."""
        if self._bm25 is None or not self._documents:
            return []

        tokenized_query = self._tokenize(query_text)
        scores = self._bm25.get_scores(tokenized_query)

        # Get top-n indices sorted by score
        indexed_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        results = []
        max_score = indexed_scores[0][1] if indexed_scores else 1.0
        max_score = max(max_score, 1e-6)  # avoid division by zero

        for idx, score in indexed_scores[:n_results]:
            if score > 0:
                normalized_score = score / max_score
                results.append((self._documents[idx], normalized_score))

        return results

    @property
    def count(self) -> int:
        return len(self._documents)
