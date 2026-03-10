"""Reciprocal Rank Fusion (RRF) with adaptive weights."""

from __future__ import annotations

import logging

from aris.core.models import ScoredDocument

logger = logging.getLogger(__name__)

# RRF constant (standard value from the original paper)
RRF_K = 60


def reciprocal_rank_fusion(
    result_lists: list[list[ScoredDocument]],
    weights: list[float] | None = None,
    max_results: int = 100,
) -> list[ScoredDocument]:
    """Fuse multiple ranked result lists using weighted Reciprocal Rank Fusion.

    Args:
        result_lists: List of ranked result lists from different retrievers.
        weights: Per-list weights (default: equal weights).
        max_results: Maximum number of results to return.

    Returns:
        Fused, re-ranked list of ScoredDocuments.
    """
    if not result_lists:
        return []

    if weights is None:
        weights = [1.0] * len(result_lists)

    # Normalize weights
    total_weight = sum(weights)
    if total_weight > 0:
        weights = [w / total_weight for w in weights]

    # Compute RRF scores
    url_scores: dict[str, float] = {}
    url_docs: dict[str, ScoredDocument] = {}

    for list_idx, results in enumerate(result_lists):
        weight = weights[list_idx] if list_idx < len(weights) else 1.0
        for rank, doc in enumerate(results):
            url = doc.document.url
            rrf_score = weight / (RRF_K + rank + 1)
            url_scores[url] = url_scores.get(url, 0) + rrf_score

            # Keep the doc with the highest original score
            if url not in url_docs or doc.retrieval_score > url_docs[url].retrieval_score:
                url_docs[url] = doc

    # Sort by RRF score
    sorted_urls = sorted(url_scores.keys(), key=lambda u: url_scores[u], reverse=True)

    fused = []
    for url in sorted_urls[:max_results]:
        doc = url_docs[url]
        doc.retrieval_score = url_scores[url]
        doc.final_score = url_scores[url]
        fused.append(doc)

    # Normalize scores to [0, 1]
    if fused:
        max_score = fused[0].retrieval_score
        if max_score > 0:
            for doc in fused:
                doc.retrieval_score /= max_score
                doc.final_score /= max_score

    return fused
