"""Predicts which failure modes a query is likely to trigger.

Uses SearchProbe's FailureMode taxonomy to drive strategy selection.
"""

from __future__ import annotations

import re

from aris.que.models import StructuredQuery

# Import SearchProbe taxonomy
try:
    from searchprobe.intelligence.taxonomy import FAILURE_ROOT_CAUSES, FailureMode
except ImportError:
    # Standalone mode without SearchProbe
    from enum import Enum

    class FailureMode(str, Enum):
        NEGATION_COLLAPSE = "negation_collapse"
        NUMERIC_BLINDNESS = "numeric_blindness"
        TEMPORAL_INSENSITIVITY = "temporal_insensitivity"
        SEMANTIC_CONFLATION = "semantic_conflation"
        KEYWORD_LEAKAGE = "keyword_leakage"
        POPULARITY_BIAS = "popularity_bias"
        DOMAIN_MISMATCH = "domain_mismatch"
        PARTIAL_CONSTRAINT = "partial_constraint"
        INVERSE_RESULT = "inverse_result"
        TANGENTIAL_RESULT = "tangential_result"
        NO_RESULTS = "no_results"
        SEARCH_FAILED = "search_failed"

    FAILURE_ROOT_CAUSES = {}


# Query feature -> likely failure modes
_FEATURE_TO_MODES: dict[str, list[str]] = {
    "has_negation": [
        FailureMode.NEGATION_COLLAPSE.value,
        FailureMode.INVERSE_RESULT.value,
    ],
    "has_numeric": [
        FailureMode.NUMERIC_BLINDNESS.value,
        FailureMode.PARTIAL_CONSTRAINT.value,
    ],
    "has_temporal": [
        FailureMode.TEMPORAL_INSENSITIVITY.value,
        FailureMode.PARTIAL_CONSTRAINT.value,
    ],
    "has_entities": [
        FailureMode.SEMANTIC_CONFLATION.value,
        FailureMode.POPULARITY_BIAS.value,
    ],
    "multi_constraint": [
        FailureMode.PARTIAL_CONSTRAINT.value,
    ],
    "complex_query": [
        FailureMode.KEYWORD_LEAKAGE.value,
        FailureMode.TANGENTIAL_RESULT.value,
    ],
}


def predict_failure_modes(query: StructuredQuery) -> list[str]:
    """Predict which failure modes this query is likely to trigger."""
    modes: set[str] = set()

    if query.negations:
        modes.update(_FEATURE_TO_MODES["has_negation"])

    has_numeric = any(c.type.value == "numeric" for c in query.constraints)
    if has_numeric:
        modes.update(_FEATURE_TO_MODES["has_numeric"])

    has_temporal = any(c.type.value == "temporal" for c in query.constraints)
    if has_temporal:
        modes.update(_FEATURE_TO_MODES["has_temporal"])

    if query.entities:
        modes.update(_FEATURE_TO_MODES["has_entities"])

    if len(query.constraints) >= 3:
        modes.update(_FEATURE_TO_MODES["multi_constraint"])

    # Complex query heuristic: long query with multiple clauses
    if len(query.original_query.split()) > 15 or "," in query.original_query:
        modes.update(_FEATURE_TO_MODES["complex_query"])

    return sorted(modes)
