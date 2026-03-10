"""Post-retrieval negation filtering.

Scans each result for negated terms and removes matches.
This is the architectural solution to negation collapse:
negations are stripped before embedding, then filtered here.
"""

from __future__ import annotations

import logging
import re

from aris.core.models import ScoredDocument

logger = logging.getLogger(__name__)


def filter_negations(
    documents: list[ScoredDocument],
    negated_terms: list[str],
) -> list[ScoredDocument]:
    """Remove documents that mention any negated term.

    Args:
        documents: Candidate documents from retrieval.
        negated_terms: Terms that must NOT appear in results.

    Returns:
        Filtered list with negation-violating documents removed.
    """
    if not negated_terms:
        return documents

    filtered = []
    removed_count = 0

    for doc in documents:
        text = _get_searchable_text(doc)
        if _contains_negated_term(text, negated_terms):
            removed_count += 1
            continue
        filtered.append(doc)

    if removed_count > 0:
        logger.info(
            "Negation filter: removed %d/%d documents matching %s",
            removed_count, len(documents), negated_terms,
        )

    return filtered


def _get_searchable_text(doc: ScoredDocument) -> str:
    """Get all searchable text from a document."""
    parts = [
        doc.document.title,
        doc.document.snippet,
        doc.document.content or "",
    ]
    return " ".join(parts).lower()


def _contains_negated_term(text: str, negated_terms: list[str]) -> bool:
    """Check if text contains any negated term."""
    for term in negated_terms:
        # Use word boundary matching to avoid partial matches
        pattern = re.compile(r"\b" + re.escape(term.lower()) + r"\b", re.IGNORECASE)
        if pattern.search(text):
            return True
    return False
