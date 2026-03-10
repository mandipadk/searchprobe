"""Confidence calibration for search results."""

from __future__ import annotations

from aris.core.models import ScoredDocument


def calibrate_confidence(candidates: list[ScoredDocument]) -> list[ScoredDocument]:
    """Compute and calibrate confidence scores for results.

    Confidence reflects how certain we are that the result is correct,
    considering both relevance score and constraint verification.
    """
    for doc in candidates:
        # Combine signals
        signals = []

        # Retrieval score confidence
        if doc.retrieval_score > 0.7:
            signals.append(0.9)
        elif doc.retrieval_score > 0.4:
            signals.append(0.6)
        else:
            signals.append(0.3)

        # Semantic score confidence (if cross-encoder was run)
        if doc.semantic_score > 0:
            if doc.semantic_score > 0.7:
                signals.append(0.95)
            elif doc.semantic_score > 0.4:
                signals.append(0.7)
            else:
                signals.append(0.4)

        # Verification confidence
        if doc.constraint_results:
            total = len(doc.constraint_results)
            satisfied = sum(1 for v in doc.constraint_results.values() if v)
            signals.append(satisfied / total if total > 0 else 0.5)

        doc.confidence = sum(signals) / len(signals) if signals else 0.5

    return candidates
