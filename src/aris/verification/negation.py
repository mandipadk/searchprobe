"""Negation constraint verifier.

Scans result content for negated concepts and flags violations.
"""

from __future__ import annotations

import re

from aris.core.models import Document
from aris.verification.models import ConstraintStatus, VerificationResult


def verify_negation(document: Document, negated_terms: list[str]) -> VerificationResult:
    """Verify that a document does NOT mention any negated terms.

    This is the verification counterpart to retrieval/negation_filter.py.
    While the filter removes obvious matches, the verifier does a deeper check.
    """
    text = f"{document.title} {document.snippet} {document.content or ''}".lower()

    for term in negated_terms:
        pattern = re.compile(r"\b" + re.escape(term.lower()) + r"\b")
        match = pattern.search(text)
        if match:
            # Find context around the match
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 50)
            context = text[start:end]

            return VerificationResult(
                constraint_type="negation",
                constraint_description=f"Must NOT mention: {term}",
                status=ConstraintStatus.VIOLATED,
                confidence=0.95,
                evidence=f"Found '{term}' in: ...{context}...",
            )

    return VerificationResult(
        constraint_type="negation",
        constraint_description=f"Must NOT mention: {', '.join(negated_terms)}",
        status=ConstraintStatus.SATISFIED,
        confidence=0.9,
    )
