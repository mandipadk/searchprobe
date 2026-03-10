"""Ranking engine -- orchestrates multi-dimensional scoring and reranking."""

from __future__ import annotations

import logging

from aris.core.config import ArisConfig
from aris.core.models import ScoredDocument
from aris.que.models import StructuredQuery
from aris.ranking.confidence import calibrate_confidence
from aris.ranking.constraint_scorer import apply_constraint_penalty
from aris.ranking.cross_encoder import CrossEncoderReranker
from aris.ranking.diversity import mmr_diversify
from aris.ranking.score_fusion import fuse_scores

logger = logging.getLogger(__name__)


class RankingEngine:
    """Orchestrates ranking: cross-encoder, constraint scoring, fusion, diversity."""

    def __init__(self, config: ArisConfig, use_cross_encoder: bool = False) -> None:
        self._config = config
        self._cross_encoder = CrossEncoderReranker(config) if use_cross_encoder else None

    async def rank(
        self,
        query: StructuredQuery,
        candidates: list[ScoredDocument],
        num_results: int | None = None,
    ) -> list[ScoredDocument]:
        """Rank candidates and return top results."""
        num_results = num_results or self._config.default_num_results

        if not candidates:
            return []

        # 1. Cross-encoder reranking (if available)
        if self._cross_encoder:
            search_query = query.core_intent or query.original_query
            candidates = await self._cross_encoder.rerank(search_query, candidates)

        # 2. Apply constraint penalty (AND-logic)
        candidates = apply_constraint_penalty(candidates)

        # 3. Fuse scores
        for doc in candidates:
            doc.final_score = fuse_scores(doc)

        # 4. Sort by final score
        candidates.sort(key=lambda d: d.final_score, reverse=True)

        # 5. Calibrate confidence
        candidates = calibrate_confidence(candidates)

        # 6. Apply MMR diversity
        candidates = mmr_diversify(candidates, num_results=num_results)

        return candidates[:num_results]
