"""Evaluation engine for search result quality assessment."""

from searchprobe.evaluation.dimensions import (
    DIMENSION_WEIGHTS,
    EvaluationDimension,
    calculate_weighted_score,
    get_weights_for_category,
)
from searchprobe.evaluation.judge import EvaluationResult, SearchJudge

__all__ = [
    "EvaluationDimension",
    "DIMENSION_WEIGHTS",
    "get_weights_for_category",
    "calculate_weighted_score",
    "SearchJudge",
    "EvaluationResult",
]
