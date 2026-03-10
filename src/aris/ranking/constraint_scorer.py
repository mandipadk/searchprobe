"""Constraint-aware scoring with AND-logic penalty."""

from __future__ import annotations

from aris.core.models import ScoredDocument


def apply_constraint_penalty(candidates: list[ScoredDocument]) -> list[ScoredDocument]:
    """Apply constraint-aware penalty to candidate scores.

    Uses AND-logic: partial constraint satisfaction is heavily penalized.
    A document that satisfies 4/5 constraints is much worse than one satisfying 5/5,
    because the user asked for ALL constraints.
    """
    for doc in candidates:
        if not doc.constraint_results:
            continue

        total = len(doc.constraint_results)
        satisfied = sum(1 for v in doc.constraint_results.values() if v)

        if total == 0:
            continue

        ratio = satisfied / total

        # AND-logic penalty: quadratic to heavily penalize partial satisfaction
        # ratio=1.0 -> penalty=1.0 (no penalty)
        # ratio=0.8 -> penalty=0.64
        # ratio=0.5 -> penalty=0.25
        # ratio=0.0 -> penalty=0.0
        constraint_score = ratio ** 2
        doc.verification_score = constraint_score

    return candidates
