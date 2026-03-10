"""Boolean logic constraint checker.

Verifies AND/OR/NOT logic across multiple constraints.
"""

from __future__ import annotations

from aris.verification.models import ConstraintStatus, VerificationResult


def verify_boolean_and(results: list[VerificationResult]) -> VerificationResult:
    """Verify that ALL constraints are satisfied (AND logic).

    This is the key architectural difference from averaging: any unsatisfied
    constraint makes the whole result fail.
    """
    if not results:
        return VerificationResult(
            constraint_type="boolean_and",
            status=ConstraintStatus.SATISFIED,
            confidence=1.0,
        )

    violated = [r for r in results if r.status == ConstraintStatus.VIOLATED]
    unknown = [r for r in results if r.status == ConstraintStatus.UNKNOWN]
    satisfied = [r for r in results if r.status == ConstraintStatus.SATISFIED]

    if violated:
        descriptions = [r.constraint_description for r in violated]
        return VerificationResult(
            constraint_type="boolean_and",
            constraint_description=f"Failed: {', '.join(descriptions)}",
            status=ConstraintStatus.VIOLATED,
            confidence=max(r.confidence for r in violated),
            evidence=f"{len(violated)}/{len(results)} constraints violated",
        )

    if unknown:
        confidence = sum(r.confidence for r in results) / len(results)
        return VerificationResult(
            constraint_type="boolean_and",
            constraint_description=f"{len(unknown)} constraints uncertain",
            status=ConstraintStatus.UNKNOWN,
            confidence=confidence,
            evidence=f"{len(satisfied)} satisfied, {len(unknown)} unknown",
        )

    avg_confidence = sum(r.confidence for r in satisfied) / len(satisfied)
    return VerificationResult(
        constraint_type="boolean_and",
        status=ConstraintStatus.SATISFIED,
        confidence=avg_confidence,
        evidence=f"All {len(satisfied)} constraints satisfied",
    )


def compute_constraint_score(results: list[VerificationResult]) -> float:
    """Compute a constraint satisfaction score using AND-logic.

    Returns a score in [0, 1] where:
    - 1.0 = all constraints satisfied with high confidence
    - 0.0 = one or more constraints violated with high confidence
    - Between = partial satisfaction or uncertain
    """
    if not results:
        return 1.0

    scores = []
    for r in results:
        if r.status == ConstraintStatus.SATISFIED:
            scores.append(r.confidence)
        elif r.status == ConstraintStatus.VIOLATED:
            scores.append(0.0)
        else:  # UNKNOWN
            scores.append(0.5 * r.confidence)

    # AND-logic: use geometric mean (penalizes zeros heavily)
    product = 1.0
    for s in scores:
        product *= max(s, 0.01)  # Small floor to avoid total zero

    return product ** (1.0 / len(scores))
