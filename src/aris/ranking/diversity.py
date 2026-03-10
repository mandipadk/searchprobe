"""Result diversification using Maximal Marginal Relevance (MMR)."""

from __future__ import annotations

from aris.core.models import ScoredDocument


def mmr_diversify(
    candidates: list[ScoredDocument],
    num_results: int = 10,
    lambda_param: float = 0.7,
) -> list[ScoredDocument]:
    """Apply Maximal Marginal Relevance to diversify results.

    Balances relevance with diversity to prevent redundant results.

    Args:
        candidates: Ranked candidate documents.
        num_results: Number of results to return.
        lambda_param: Balance between relevance (1.0) and diversity (0.0).
    """
    if len(candidates) <= num_results:
        return candidates

    selected: list[ScoredDocument] = []
    remaining = list(candidates)

    while len(selected) < num_results and remaining:
        best_idx = 0
        best_mmr_score = float("-inf")

        for i, candidate in enumerate(remaining):
            # Relevance component
            relevance = candidate.final_score

            # Diversity component: max similarity to already selected
            max_similarity = 0.0
            for sel in selected:
                sim = _text_similarity(candidate, sel)
                max_similarity = max(max_similarity, sim)

            # MMR score
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity

            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_idx = i

        selected.append(remaining.pop(best_idx))

    return selected


def _text_similarity(a: ScoredDocument, b: ScoredDocument) -> float:
    """Simple token overlap similarity between two documents."""
    text_a = set((a.document.title + " " + a.document.snippet).lower().split())
    text_b = set((b.document.title + " " + b.document.snippet).lower().split())

    if not text_a or not text_b:
        return 0.0

    intersection = text_a & text_b
    union = text_a | text_b
    return len(intersection) / len(union) if union else 0.0
