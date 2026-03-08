"""Structured failure mode taxonomy for adversarial search analysis.

Classifies free-text failure descriptions from the LLM judge into structured
FailureMode enums with root cause mapping, severity, and category associations.
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class FailureMode(str, Enum):
    """Structured failure modes with root cause grounding."""

    # Embedding-level failures
    NEGATION_COLLAPSE = "negation_collapse"
    NUMERIC_BLINDNESS = "numeric_blindness"
    TEMPORAL_INSENSITIVITY = "temporal_insensitivity"
    SEMANTIC_CONFLATION = "semantic_conflation"

    # Retrieval-level failures
    KEYWORD_LEAKAGE = "keyword_leakage"
    POPULARITY_BIAS = "popularity_bias"
    DOMAIN_MISMATCH = "domain_mismatch"

    # Result-level failures
    PARTIAL_CONSTRAINT = "partial_constraint"
    INVERSE_RESULT = "inverse_result"
    TANGENTIAL_RESULT = "tangential_result"

    # Meta failures
    NO_RESULTS = "no_results"
    SEARCH_FAILED = "search_failed"


FAILURE_ROOT_CAUSES: dict[FailureMode, dict[str, Any]] = {
    FailureMode.NEGATION_COLLAPSE: {
        "description": "Embedding space collapses negated and non-negated forms",
        "severity": "high",
        "categories": ["negation", "antonym_confusion", "boolean_logic"],
        "indicators": [
            "not", "never", "no", "without", "non-", "neither", "nor",
            "violating the negation", "opposite", "ignoring the exclusion",
            "directly contradicts", "negation constraint",
        ],
    },
    FailureMode.NUMERIC_BLINDNESS: {
        "description": "Embeddings treat numbers as tokens, not values",
        "severity": "high",
        "categories": ["numeric_precision", "multi_constraint"],
        "indicators": [
            "numeric", "number", "exactly", "between", "range",
            "greater than", "less than", "more than", "fewer than",
            "wrong number", "incorrect value",
        ],
    },
    FailureMode.TEMPORAL_INSENSITIVITY: {
        "description": "Embeddings encode time references weakly",
        "severity": "medium",
        "categories": ["temporal_constraint"],
        "indicators": [
            "date", "time", "before", "after", "since", "until",
            "week", "month", "year", "outdated", "wrong period",
            "temporal", "not recent",
        ],
    },
    FailureMode.SEMANTIC_CONFLATION: {
        "description": "Multiple meanings of a term are blended in embedding space",
        "severity": "medium",
        "categories": ["polysemy", "entity_disambiguation"],
        "indicators": [
            "wrong sense", "ambiguous", "different meaning",
            "confused with", "mixed up", "wrong entity",
            "disambiguation", "multiple meanings",
        ],
    },
    FailureMode.KEYWORD_LEAKAGE: {
        "description": "Results match keywords but not intent",
        "severity": "medium",
        "categories": ["instruction_following", "compositional"],
        "indicators": [
            "keyword", "mentions but", "contains the word",
            "surface match", "lexical", "not relevant",
            "only mentions", "tangentially related",
        ],
    },
    FailureMode.POPULARITY_BIAS: {
        "description": "Popular results override relevance",
        "severity": "low",
        "categories": ["specificity_gradient", "entity_disambiguation"],
        "indicators": [
            "popular", "well-known", "famous", "dominant",
            "most common", "mainstream", "generic",
        ],
    },
    FailureMode.DOMAIN_MISMATCH: {
        "description": "Wrong content type returned",
        "severity": "low",
        "categories": ["instruction_following"],
        "indicators": [
            "wrong type", "documentation instead", "blog instead",
            "forum instead", "not academic", "not a paper",
            "wrong format", "wrong source type",
        ],
    },
    FailureMode.PARTIAL_CONSTRAINT: {
        "description": "Some constraints satisfied but not all",
        "severity": "high",
        "categories": ["multi_constraint", "boolean_logic"],
        "indicators": [
            "partial", "some constraints", "only satisfies",
            "missing constraint", "not all", "one of",
            "but not", "satisfies only",
        ],
    },
    FailureMode.INVERSE_RESULT: {
        "description": "Result is the opposite of what was asked",
        "severity": "high",
        "categories": ["negation", "antonym_confusion", "counterfactual"],
        "indicators": [
            "opposite", "inverse", "reversed", "contrary",
            "directly violating", "antonym", "wrong polarity",
        ],
    },
    FailureMode.TANGENTIAL_RESULT: {
        "description": "Related topic but wrong answer",
        "severity": "low",
        "categories": ["compositional", "specificity_gradient", "cross_lingual"],
        "indicators": [
            "tangential", "related but", "adjacent", "nearby topic",
            "not directly", "loosely related", "off-topic",
        ],
    },
    FailureMode.NO_RESULTS: {
        "description": "No results returned",
        "severity": "high",
        "categories": [],
        "indicators": ["no results", "empty"],
    },
    FailureMode.SEARCH_FAILED: {
        "description": "Search API error",
        "severity": "high",
        "categories": [],
        "indicators": ["search_failed", "error", "timeout"],
    },
}


class FailureClassifier:
    """Classifies free-text failure descriptions into structured FailureModes.

    Uses keyword matching with category-aware boosting. Designed to work with
    the free-text failure_modes field from EvaluationResult.
    """

    def __init__(self) -> None:
        # Pre-compute lowercase indicators for each mode
        self._indicator_map: dict[FailureMode, list[str]] = {
            mode: [ind.lower() for ind in info["indicators"]]
            for mode, info in FAILURE_ROOT_CAUSES.items()
        }
        self._category_map: dict[FailureMode, list[str]] = {
            mode: info["categories"]
            for mode, info in FAILURE_ROOT_CAUSES.items()
        }

    def classify(self, failure_text: str, category: str = "") -> list[FailureMode]:
        """Classify a single failure description.

        Args:
            failure_text: Free-text failure description from LLM judge.
            category: Adversarial category (used for boosting).

        Returns:
            Ranked list of matching FailureModes (best match first).
        """
        text_lower = failure_text.lower()
        scores: dict[FailureMode, float] = {}

        for mode, indicators in self._indicator_map.items():
            score = 0.0
            for indicator in indicators:
                if indicator in text_lower:
                    score += 1.0

            # Category boost: if the failure mode is associated with this category,
            # add 0.5 to the score
            if category and category in self._category_map.get(mode, []):
                score += 0.5

            if score > 0:
                scores[mode] = score

        # Sort by score descending
        ranked = sorted(scores.keys(), key=lambda m: scores[m], reverse=True)
        return ranked

    def classify_evaluation(self, eval_result: dict[str, Any]) -> list[FailureMode]:
        """Classify all failure modes from an evaluation result.

        Args:
            eval_result: Evaluation result dict with 'failure_modes' and 'category'.

        Returns:
            Deduplicated, ranked list of FailureModes.
        """
        category = eval_result.get("category", "")
        failure_texts = eval_result.get("failure_modes", [])

        all_modes: dict[FailureMode, float] = {}
        for text in failure_texts:
            if isinstance(text, str):
                for mode in self.classify(text, category):
                    all_modes[mode] = all_modes.get(mode, 0) + 1

        return sorted(all_modes.keys(), key=lambda m: all_modes[m], reverse=True)

    def get_root_cause(self, mode: FailureMode) -> dict[str, Any]:
        """Get root cause information for a failure mode."""
        return FAILURE_ROOT_CAUSES.get(mode, {})
