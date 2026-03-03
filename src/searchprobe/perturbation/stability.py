"""Stability metrics for measuring result set consistency under perturbation.

Implements Jaccard similarity and Rank-Biased Overlap (Webber et al. 2010).
"""

import math

import numpy as np

from searchprobe.perturbation.models import PerturbationAnalysis, SensitivityMap
from searchprobe.perturbation.operators import PerturbationType


def jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Compute Jaccard similarity between two result sets.

    J(A, B) = |A ∩ B| / |A ∪ B|

    Args:
        set_a: First result set (URLs)
        set_b: Second result set (URLs)

    Returns:
        Jaccard similarity in [0, 1]
    """
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0

    intersection = len(set_a & set_b)
    union = len(set_a | set_b)

    return intersection / union if union > 0 else 0.0


def rank_biased_overlap(
    list_a: list[str],
    list_b: list[str],
    p: float = 0.9,
) -> float:
    """Compute Rank-Biased Overlap between two ranked lists.

    RBO (Webber et al. 2010) is a top-weighted similarity measure for
    indefinite ranked lists. The parameter p controls the steepness of
    the top-weighting (higher p = more weight on deeper ranks).

    Args:
        list_a: First ranked list (URLs in order)
        list_b: Second ranked list (URLs in order)
        p: Persistence parameter in (0, 1). Default 0.9 weights top-10 heavily.

    Returns:
        RBO score in [0, 1]
    """
    if not list_a or not list_b:
        return 0.0

    k = min(len(list_a), len(list_b))
    if k == 0:
        return 0.0

    # Compute overlap at each depth
    rbo = 0.0
    set_a: set[str] = set()
    set_b: set[str] = set()

    for d in range(1, k + 1):
        set_a.add(list_a[d - 1])
        set_b.add(list_b[d - 1])

        overlap = len(set_a & set_b)
        agreement = overlap / d

        rbo += (p ** (d - 1)) * agreement

    # Normalize
    rbo *= (1 - p)

    return min(1.0, rbo)


def compute_sensitivity_map(
    query: str,
    analyses: list[PerturbationAnalysis],
) -> SensitivityMap:
    """Compute word-level sensitivity from perturbation analyses.

    Each word's sensitivity score reflects how much removing/changing it
    disrupts the search results. Higher = more load-bearing.

    Args:
        query: Original query
        analyses: List of PerturbationAnalysis from word_delete perturbations

    Returns:
        SensitivityMap with per-word scores
    """
    words = query.split()
    word_scores: dict[str, float] = {}

    # Default: all words start at 0 sensitivity
    for word in words:
        word_scores[word] = 0.0

    # For word_delete perturbations, sensitivity = 1 - stability
    for analysis in analyses:
        if analysis.perturbation_type == PerturbationType.WORD_DELETE.value:
            detail = analysis.perturbation_detail
            # Extract deleted word from detail string
            if "deleted '" in detail:
                deleted_word = detail.split("deleted '")[1].split("'")[0]
                # Sensitivity = how much results changed (1 - jaccard)
                sensitivity = 1.0 - analysis.jaccard_similarity
                if deleted_word in word_scores:
                    word_scores[deleted_word] = max(
                        word_scores[deleted_word], sensitivity
                    )

    # Normalize to [0, 1]
    max_score = max(word_scores.values()) if word_scores else 1.0
    if max_score > 0:
        word_scores = {w: s / max_score for w, s in word_scores.items()}

    provider = analyses[0].provider if analyses else "unknown"
    category = analyses[0].category if analyses else "unknown"

    return SensitivityMap(
        query=query,
        word_scores=word_scores,
        provider=provider,
        category=category,
    )
