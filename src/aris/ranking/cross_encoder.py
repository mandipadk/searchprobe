"""Cross-encoder reranking using sentence-transformers CrossEncoder."""

from __future__ import annotations

import logging

from aris.core.config import ArisConfig
from aris.core.models import ScoredDocument

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Reranks documents using a cross-encoder model for precise relevance scoring."""

    def __init__(self, config: ArisConfig) -> None:
        self._model_name = config.cross_encoder_model
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)
        except ImportError:
            logger.warning("sentence-transformers not installed, cross-encoder reranking disabled")

    async def rerank(
        self, query: str, candidates: list[ScoredDocument]
    ) -> list[ScoredDocument]:
        """Score each (query, document) pair with the cross-encoder."""
        self._ensure_model()
        if self._model is None or not candidates:
            return candidates

        pairs = [
            (query, (doc.document.content or doc.document.snippet)[:512])
            for doc in candidates
        ]

        try:
            scores = self._model.predict(
                [list(p) for p in pairs],
                show_progress_bar=False,
            )

            # Normalize scores to [0, 1] using sigmoid-like scaling
            min_score = min(scores)
            max_score = max(scores)
            score_range = max_score - min_score if max_score > min_score else 1.0

            for i, doc in enumerate(candidates):
                normalized = (scores[i] - min_score) / score_range
                doc.semantic_score = float(normalized)

        except Exception as e:
            logger.warning("Cross-encoder scoring failed: %s", e)

        return candidates
