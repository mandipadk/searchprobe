"""Evaluation engine for search result quality assessment."""

from searchprobe.evaluation.dimensions import (
    EvaluationDimension,
    DIMENSION_WEIGHTS,
    get_weights_for_category,
    calculate_weighted_score,
)
from searchprobe.evaluation.judge import SearchJudge, EvaluationResult

__all__ = [
    "EvaluationDimension",
    "DIMENSION_WEIGHTS",
    "get_weights_for_category",
    "calculate_weighted_score",
    "SearchJudge",
    "EvaluationResult",
]
