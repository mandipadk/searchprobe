"""Evaluation dimensions for scoring search results."""

from enum import Enum
from typing import Any

from searchprobe.queries.taxonomy import AdversarialCategory


class EvaluationDimension(str, Enum):
    """Dimensions for evaluating search result quality."""

    # Core dimensions
    RELEVANCE = "relevance"  # Overall topical relevance to query
    CONSTRAINT_SATISFACTION = "constraint_satisfaction"  # Respects explicit constraints
    ACCURACY = "accuracy"  # Factual correctness of results

    # Category-specific dimensions
    NEGATION_RESPECT = "negation_respect"  # Excludes what was negated
    NUMERIC_PRECISION = "numeric_precision"  # Matches exact numbers
    TEMPORAL_ACCURACY = "temporal_accuracy"  # Correct time period
    ENTITY_CORRECTNESS = "entity_correctness"  # Right entity disambiguation
    INSTRUCTION_COMPLIANCE = "instruction_compliance"  # Follows meta-instructions


# Default weights for dimensions
DEFAULT_WEIGHTS: dict[EvaluationDimension, float] = {
    EvaluationDimension.RELEVANCE: 0.50,
    EvaluationDimension.CONSTRAINT_SATISFACTION: 0.30,
    EvaluationDimension.ACCURACY: 0.20,
}

# Category-specific dimension weights
# These override defaults for specific adversarial categories
CATEGORY_WEIGHTS: dict[AdversarialCategory, dict[EvaluationDimension, float]] = {
    AdversarialCategory.NEGATION: {
        EvaluationDimension.RELEVANCE: 0.30,
        EvaluationDimension.NEGATION_RESPECT: 0.50,
        EvaluationDimension.CONSTRAINT_SATISFACTION: 0.20,
    },
    AdversarialCategory.NUMERIC_PRECISION: {
        EvaluationDimension.RELEVANCE: 0.30,
        EvaluationDimension.NUMERIC_PRECISION: 0.50,
        EvaluationDimension.CONSTRAINT_SATISFACTION: 0.20,
    },
    AdversarialCategory.TEMPORAL_CONSTRAINT: {
        EvaluationDimension.RELEVANCE: 0.30,
        EvaluationDimension.TEMPORAL_ACCURACY: 0.50,
        EvaluationDimension.CONSTRAINT_SATISFACTION: 0.20,
    },
    AdversarialCategory.MULTI_CONSTRAINT: {
        EvaluationDimension.RELEVANCE: 0.20,
        EvaluationDimension.CONSTRAINT_SATISFACTION: 0.60,
        EvaluationDimension.ACCURACY: 0.20,
    },
    AdversarialCategory.POLYSEMY: {
        EvaluationDimension.RELEVANCE: 0.40,
        EvaluationDimension.ENTITY_CORRECTNESS: 0.40,
        EvaluationDimension.ACCURACY: 0.20,
    },
    AdversarialCategory.COMPOSITIONAL: {
        EvaluationDimension.RELEVANCE: 0.40,
        EvaluationDimension.CONSTRAINT_SATISFACTION: 0.40,
        EvaluationDimension.ACCURACY: 0.20,
    },
    AdversarialCategory.ANTONYM_CONFUSION: {
        EvaluationDimension.RELEVANCE: 0.30,
        EvaluationDimension.CONSTRAINT_SATISFACTION: 0.50,
        EvaluationDimension.ACCURACY: 0.20,
    },
    AdversarialCategory.SPECIFICITY_GRADIENT: {
        EvaluationDimension.RELEVANCE: 0.50,
        EvaluationDimension.CONSTRAINT_SATISFACTION: 0.30,
        EvaluationDimension.ACCURACY: 0.20,
    },
    AdversarialCategory.CROSS_LINGUAL: {
        EvaluationDimension.RELEVANCE: 0.50,
        EvaluationDimension.ACCURACY: 0.30,
        EvaluationDimension.CONSTRAINT_SATISFACTION: 0.20,
    },
    AdversarialCategory.COUNTERFACTUAL: {
        EvaluationDimension.RELEVANCE: 0.30,
        EvaluationDimension.CONSTRAINT_SATISFACTION: 0.50,
        EvaluationDimension.ACCURACY: 0.20,
    },
    AdversarialCategory.BOOLEAN_LOGIC: {
        EvaluationDimension.RELEVANCE: 0.20,
        EvaluationDimension.CONSTRAINT_SATISFACTION: 0.60,
        EvaluationDimension.ACCURACY: 0.20,
    },
    AdversarialCategory.ENTITY_DISAMBIGUATION: {
        EvaluationDimension.RELEVANCE: 0.30,
        EvaluationDimension.ENTITY_CORRECTNESS: 0.50,
        EvaluationDimension.ACCURACY: 0.20,
    },
    AdversarialCategory.INSTRUCTION_FOLLOWING: {
        EvaluationDimension.RELEVANCE: 0.30,
        EvaluationDimension.INSTRUCTION_COMPLIANCE: 0.50,
        EvaluationDimension.CONSTRAINT_SATISFACTION: 0.20,
    },
}


def get_weights_for_category(
    category: AdversarialCategory | str,
) -> dict[EvaluationDimension, float]:
    """Get dimension weights for a specific adversarial category.

    Args:
        category: The adversarial category

    Returns:
        Dictionary mapping dimensions to weights (sum to 1.0)
    """
    if isinstance(category, str):
        try:
            category = AdversarialCategory(category)
        except ValueError:
            return DEFAULT_WEIGHTS

    return CATEGORY_WEIGHTS.get(category, DEFAULT_WEIGHTS)


def get_active_dimensions(
    category: AdversarialCategory | str,
) -> list[EvaluationDimension]:
    """Get the active dimensions for a category.

    Args:
        category: The adversarial category

    Returns:
        List of dimensions to evaluate for this category
    """
    weights = get_weights_for_category(category)
    return list(weights.keys())


def calculate_weighted_score(
    scores: dict[EvaluationDimension, float],
    category: AdversarialCategory | str,
) -> float:
    """Calculate weighted overall score from dimension scores.

    Args:
        scores: Dictionary of dimension -> score (0-1)
        category: The adversarial category for weight selection

    Returns:
        Weighted average score (0-1)
    """
    weights = get_weights_for_category(category)

    total_weight = 0.0
    weighted_sum = 0.0

    for dimension, weight in weights.items():
        if dimension in scores:
            weighted_sum += scores[dimension] * weight
            total_weight += weight

    if total_weight == 0:
        return 0.0

    return weighted_sum / total_weight


# For backwards compatibility
DIMENSION_WEIGHTS = DEFAULT_WEIGHTS
