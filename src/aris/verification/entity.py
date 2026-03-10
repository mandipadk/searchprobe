"""Entity disambiguation verifier.

Checks that the correct entity is referenced (right Michael Jordan, etc.)
"""

from __future__ import annotations

import re

from aris.core.models import Document
from aris.que.models import EntityReference
from aris.verification.models import ConstraintStatus, VerificationResult


def verify_entity(document: Document, entity: EntityReference) -> VerificationResult:
    """Verify that a document references the correct entity."""
    text = f"{document.title} {document.snippet} {document.content or ''}".lower()

    # Check that the entity is mentioned
    if entity.text.lower() not in text:
        return VerificationResult(
            constraint_type="entity",
            constraint_description=f"Must reference: {entity.text} ({entity.disambiguation})",
            status=ConstraintStatus.UNKNOWN,
            confidence=0.4,
            evidence=f"Entity '{entity.text}' not found in document",
        )

    # Check for disambiguation context
    if entity.disambiguation:
        disambiguation_terms = entity.disambiguation.lower().split()
        matching_terms = sum(1 for t in disambiguation_terms if t in text and len(t) > 3)
        match_ratio = matching_terms / max(len(disambiguation_terms), 1)

        if match_ratio >= 0.3:
            return VerificationResult(
                constraint_type="entity",
                constraint_description=f"Must reference: {entity.text} ({entity.disambiguation})",
                status=ConstraintStatus.SATISFIED,
                confidence=min(0.5 + match_ratio * 0.5, 0.95),
                evidence=f"Disambiguation match: {matching_terms}/{len(disambiguation_terms)} context terms found",
            )

    # Entity mentioned but disambiguation unclear
    return VerificationResult(
        constraint_type="entity",
        constraint_description=f"Must reference: {entity.text} ({entity.disambiguation})",
        status=ConstraintStatus.SATISFIED,
        confidence=0.5,
        evidence=f"Entity '{entity.text}' found but disambiguation uncertain",
    )
