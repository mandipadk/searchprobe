"""Negation detection and positive query generation."""

from __future__ import annotations

import re


# Patterns that indicate negation
_NEGATION_PATTERNS = [
    r"\bNOT\s+(\w+(?:\s+\w+)?)",
    r"\bnot\s+(\w+(?:\s+\w+)?)",
    r"\bwithout\s+(\w+(?:\s+\w+)?)",
    r"\bexcluding?\s+(\w+(?:\s+\w+)?)",
    r"\bno\s+(\w+(?:\s+\w+)?)",
    r"\bnever\s+(\w+(?:\s+\w+)?)",
    r"\bneither\s+(\w+(?:\s+\w+)?)",
    r"\bnon-(\w+)",
]

# Full negation clause patterns (capture multi-word negations)
_CLAUSE_PATTERNS = [
    r",?\s*(?:NOT|not)\s+(.+?)(?:,|$)",
    r",?\s*(?:without|excluding?)\s+(.+?)(?:,|$)",
    r",?\s*(?:but not|and not|except)\s+(.+?)(?:,|$)",
]


def detect_negations(query: str) -> list[str]:
    """Detect negated terms/concepts in a query.

    Returns a list of negated terms (e.g. ["PyTorch", "JavaScript"]).
    """
    negations = []

    # Try clause patterns first (more context)
    for pattern in _CLAUSE_PATTERNS:
        for match in re.finditer(pattern, query):
            term = match.group(1).strip().rstrip(".")
            if term and len(term) < 100:
                negations.append(term)

    # If no clause matches, try word-level patterns
    if not negations:
        for pattern in _NEGATION_PATTERNS:
            for match in re.finditer(pattern, query):
                term = match.group(1).strip()
                if term and len(term) < 50:
                    negations.append(term)

    return list(dict.fromkeys(negations))  # deduplicate, preserve order


def generate_positive_query(query: str, negations: list[str] | None = None) -> str:
    """Generate a positive version of the query for embedding search.

    Removes negation markers and negated terms so the embedding captures
    only the positive intent.
    """
    if negations is None:
        negations = detect_negations(query)

    positive = query

    # Remove negation clauses
    for pattern in _CLAUSE_PATTERNS:
        positive = re.sub(pattern, "", positive)

    # Remove negation markers with their terms
    for pattern in _NEGATION_PATTERNS:
        positive = re.sub(pattern, "", positive)

    # Remove leftover negation words
    for word in ["NOT", "not", "without", "excluding", "exclude", "never", "neither", "nor"]:
        positive = re.sub(rf"\b{word}\b", "", positive)

    # Remove the negated terms themselves
    for term in negations:
        positive = positive.replace(term, "")

    # Clean up whitespace and punctuation
    positive = re.sub(r"\s+", " ", positive).strip()
    positive = re.sub(r"^[,\s]+|[,\s]+$", "", positive)
    positive = re.sub(r",\s*,", ",", positive)

    return positive
