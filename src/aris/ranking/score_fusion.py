"""Score fusion -- combines scores from multiple ranking dimensions."""

from __future__ import annotations

from aris.core.models import ScoredDocument


def fuse_scores(
    doc: ScoredDocument,
    semantic_weight: float = 0.4,
    constraint_weight: float = 0.5,
    source_weight: float = 0.1,
) -> float:
    """Fuse multiple score dimensions into a final score.

    Args:
        doc: Document with individual scores populated.
        semantic_weight: Weight for semantic/cross-encoder score.
        constraint_weight: Weight for constraint satisfaction score.
        source_weight: Weight for source quality score.

    Returns:
        Fused score in [0, 1].
    """
    semantic = doc.semantic_score if doc.semantic_score > 0 else doc.retrieval_score
    constraint = doc.verification_score
    source_quality = _source_quality(doc.document.source)

    fused = (
        semantic * semantic_weight
        + constraint * constraint_weight
        + source_quality * source_weight
    )
    return min(1.0, max(0.0, fused))


def _source_quality(source: str) -> float:
    """Heuristic source quality score."""
    quality_map = {
        "brave": 0.8,
        "serpapi": 0.8,
        "duckduckgo": 0.7,
        "web": 0.5,
        "index": 0.6,
    }
    return quality_map.get(source, 0.5)
