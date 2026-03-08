"""Ground truth validation engine with pluggable strategies.

Provides objective, reproducible validation of search results using structured
criteria. Complements (does not replace) the LLM judge with deterministic checks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ValidationOutcome:
    """Result of applying a single validation strategy."""

    strategy: str
    passed: bool
    score: float  # 0.0 to 1.0
    matched_results: list[int] = field(default_factory=list)
    explanation: str = ""


class ValidationStrategy(Protocol):
    """Protocol for ground truth validation strategies."""

    name: str

    def validate(
        self, results: list[dict[str, Any]], criteria: dict[str, Any]
    ) -> ValidationOutcome: ...


class MustContainStrategy:
    """Results must contain specific terms/phrases."""

    name = "must_contain"

    def validate(
        self, results: list[dict[str, Any]], criteria: dict[str, Any]
    ) -> ValidationOutcome:
        terms = criteria.get("terms", [])
        if not terms:
            return ValidationOutcome(strategy=self.name, passed=True, score=1.0)

        matched: list[int] = []
        for i, result in enumerate(results):
            text = _result_text(result).lower()
            if any(term.lower() in text for term in terms):
                matched.append(i)

        score = len(matched) / max(len(results), 1)
        return ValidationOutcome(
            strategy=self.name,
            passed=score > 0.5,
            score=score,
            matched_results=matched,
            explanation=f"{len(matched)}/{len(results)} results contain required terms",
        )


class MustNotContainStrategy:
    """Results must NOT contain specific terms (negation testing)."""

    name = "must_not_contain"

    def validate(
        self, results: list[dict[str, Any]], criteria: dict[str, Any]
    ) -> ValidationOutcome:
        terms = criteria.get("terms", [])
        if not terms:
            return ValidationOutcome(strategy=self.name, passed=True, score=1.0)

        violations: list[int] = []
        for i, result in enumerate(results):
            text = _result_text(result).lower()
            if any(term.lower() in text for term in terms):
                violations.append(i)

        clean_count = len(results) - len(violations)
        score = clean_count / max(len(results), 1)
        return ValidationOutcome(
            strategy=self.name,
            passed=len(violations) == 0,
            score=score,
            matched_results=violations,
            explanation=f"{len(violations)} results contain prohibited terms",
        )


class EntityMatchStrategy:
    """Specific entities must appear in results."""

    name = "entity_match"

    def validate(
        self, results: list[dict[str, Any]], criteria: dict[str, Any]
    ) -> ValidationOutcome:
        entities = criteria.get("entities", [])
        if not entities:
            return ValidationOutcome(strategy=self.name, passed=True, score=1.0)

        all_text = " ".join(_result_text(r) for r in results).lower()
        found = [e for e in entities if e.lower() in all_text]
        score = len(found) / len(entities)
        return ValidationOutcome(
            strategy=self.name,
            passed=score >= 0.5,
            score=score,
            explanation=f"Found {len(found)}/{len(entities)} required entities",
        )


class NumericRangeStrategy:
    """Numeric values in results must be within a specified range."""

    name = "numeric_range"

    def validate(
        self, results: list[dict[str, Any]], criteria: dict[str, Any]
    ) -> ValidationOutcome:
        min_val = criteria.get("min")
        max_val = criteria.get("max")
        if min_val is None and max_val is None:
            return ValidationOutcome(strategy=self.name, passed=True, score=1.0)

        matched: list[int] = []
        for i, result in enumerate(results):
            text = _result_text(result)
            numbers = [float(n) for n in re.findall(r"[\d,]+\.?\d*", text.replace(",", "")) if n]
            for num in numbers:
                in_range = True
                if min_val is not None and num < min_val:
                    in_range = False
                if max_val is not None and num > max_val:
                    in_range = False
                if in_range:
                    matched.append(i)
                    break

        score = len(matched) / max(len(results), 1)
        return ValidationOutcome(
            strategy=self.name,
            passed=score > 0.3,
            score=score,
            matched_results=matched,
            explanation=f"{len(matched)} results contain numbers in range [{min_val}, {max_val}]",
        )


class DomainMatchStrategy:
    """Results must come from specific domains."""

    name = "domain_match"

    def validate(
        self, results: list[dict[str, Any]], criteria: dict[str, Any]
    ) -> ValidationOutcome:
        domains = [d.lower() for d in criteria.get("domains", [])]
        if not domains:
            return ValidationOutcome(strategy=self.name, passed=True, score=1.0)

        matched: list[int] = []
        for i, result in enumerate(results):
            url = str(result.get("url", "")).lower()
            if any(d in url for d in domains):
                matched.append(i)

        score = len(matched) / max(len(results), 1)
        return ValidationOutcome(
            strategy=self.name,
            passed=score > 0.3,
            score=score,
            matched_results=matched,
            explanation=f"{len(matched)} results from required domains",
        )


class PatternMatchStrategy:
    """Regex pattern matching on result text."""

    name = "pattern_match"

    def validate(
        self, results: list[dict[str, Any]], criteria: dict[str, Any]
    ) -> ValidationOutcome:
        pattern = criteria.get("pattern", "")
        if not pattern:
            return ValidationOutcome(strategy=self.name, passed=True, score=1.0)

        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error:
            return ValidationOutcome(
                strategy=self.name, passed=False, score=0.0,
                explanation=f"Invalid regex: {pattern}",
            )

        matched: list[int] = []
        for i, result in enumerate(results):
            text = _result_text(result)
            if regex.search(text):
                matched.append(i)

        score = len(matched) / max(len(results), 1)
        return ValidationOutcome(
            strategy=self.name,
            passed=score > 0.3,
            score=score,
            matched_results=matched,
            explanation=f"{len(matched)} results match pattern",
        )


# ---------------------------------------------------------------------------
# Ground Truth Engine
# ---------------------------------------------------------------------------

# Default strategy weights for computing objective scores
DEFAULT_WEIGHTS: dict[str, float] = {
    "must_contain": 1.0,
    "must_not_contain": 1.5,  # Negation violations are weighted higher
    "entity_match": 1.0,
    "numeric_range": 1.0,
    "domain_match": 0.5,
    "pattern_match": 0.8,
}


class GroundTruthEngine:
    """Orchestrates validation strategies against search results."""

    def __init__(self) -> None:
        self.strategies: dict[str, ValidationStrategy] = {
            "must_contain": MustContainStrategy(),
            "must_not_contain": MustNotContainStrategy(),
            "entity_match": EntityMatchStrategy(),
            "numeric_range": NumericRangeStrategy(),
            "domain_match": DomainMatchStrategy(),
            "pattern_match": PatternMatchStrategy(),
        }

    def validate(
        self,
        results: list[dict[str, Any]],
        ground_truth: dict[str, Any],
    ) -> dict[str, ValidationOutcome]:
        """Run applicable strategies against results.

        Args:
            results: Search results (list of dicts with title/url/snippet/content).
            ground_truth: Ground truth criteria, e.g.:
                {
                    "must_contain": {"terms": ["Python"]},
                    "must_not_contain": {"terms": ["Java"]},
                    "numeric_range": {"min": 100, "max": 500},
                }

        Returns:
            Dict of strategy_name -> ValidationOutcome.
        """
        outcomes: dict[str, ValidationOutcome] = {}

        for strategy_name, criteria in ground_truth.items():
            strategy = self.strategies.get(strategy_name)
            if strategy is None:
                continue
            if not isinstance(criteria, dict):
                criteria = {"terms": criteria} if isinstance(criteria, list) else {}
            outcomes[strategy_name] = strategy.validate(results, criteria)

        return outcomes

    def compute_objective_score(
        self,
        outcomes: dict[str, ValidationOutcome],
        weights: dict[str, float] | None = None,
    ) -> float:
        """Compute a weighted objective score from validation outcomes.

        Args:
            outcomes: Results from validate().
            weights: Optional per-strategy weights (defaults to DEFAULT_WEIGHTS).

        Returns:
            Score in [0.0, 1.0].
        """
        weights = weights or DEFAULT_WEIGHTS
        total_weight = 0.0
        weighted_sum = 0.0

        for name, outcome in outcomes.items():
            w = weights.get(name, 1.0)
            weighted_sum += outcome.score * w
            total_weight += w

        if total_weight == 0:
            return 0.0
        return weighted_sum / total_weight


def _result_text(result: dict[str, Any]) -> str:
    """Extract searchable text from a result dict."""
    parts = [
        result.get("title", ""),
        result.get("snippet", ""),
        result.get("content", "") or "",
    ]
    return " ".join(parts)
