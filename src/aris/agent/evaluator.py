"""Self-evaluation of search result quality."""

from __future__ import annotations

import logging

from aris.core.models import ScoredDocument
from aris.que.models import StructuredQuery

logger = logging.getLogger(__name__)


class ResultEvaluator:
    """Evaluates whether search results sufficiently satisfy the query constraints."""

    def __init__(
        self,
        min_results: int = 3,
        min_constraint_satisfaction: float = 0.7,
        min_avg_score: float = 0.4,
    ) -> None:
        self._min_results = min_results
        self._min_constraint_satisfaction = min_constraint_satisfaction
        self._min_avg_score = min_avg_score

    def evaluate(
        self, query: StructuredQuery, results: list[ScoredDocument]
    ) -> tuple[bool, str]:
        """Evaluate if results are good enough.

        Returns (is_sufficient, reason).
        """
        if not results:
            return False, "No results found"

        if len(results) < self._min_results:
            return False, f"Only {len(results)} results (need {self._min_results})"

        # Check average final score
        avg_score = sum(r.final_score for r in results) / len(results)
        if avg_score < self._min_avg_score:
            return False, f"Average score {avg_score:.2f} below threshold {self._min_avg_score}"

        # Check constraint satisfaction across top results
        if query.constraints or query.negations:
            top_results = results[:5]
            fully_satisfied = 0
            for r in top_results:
                if r.constraint_results:
                    if all(r.constraint_results.values()):
                        fully_satisfied += 1
                else:
                    fully_satisfied += 1  # No constraints to check

            satisfaction_ratio = fully_satisfied / len(top_results)
            if satisfaction_ratio < self._min_constraint_satisfaction:
                return False, (
                    f"Only {fully_satisfied}/{len(top_results)} top results "
                    f"fully satisfy constraints (need {self._min_constraint_satisfaction:.0%})"
                )

        return True, "Results meet quality criteria"
