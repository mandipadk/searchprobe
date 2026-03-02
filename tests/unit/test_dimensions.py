"""Tests for evaluation dimensions."""

import pytest

from searchprobe.evaluation.dimensions import (
    EvaluationDimension,
    get_weights_for_category,
    calculate_weighted_score,
    get_active_dimensions,
)
from searchprobe.queries.taxonomy import AdversarialCategory


def test_default_weights_sum_to_one():
    """Default weights should sum to 1.0."""
    weights = get_weights_for_category("unknown_category")
    total = sum(weights.values())
    assert abs(total - 1.0) < 0.01


def test_category_weights_sum_to_one():
    """All category-specific weights should sum to 1.0."""
    for category in AdversarialCategory:
        weights = get_weights_for_category(category)
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01, f"Weights for {category} sum to {total}"


def test_negation_category_weights():
    """Negation category should prioritize negation_respect dimension."""
    weights = get_weights_for_category(AdversarialCategory.NEGATION)
    assert EvaluationDimension.NEGATION_RESPECT in weights
    assert weights[EvaluationDimension.NEGATION_RESPECT] >= 0.4


def test_weighted_score_calculation():
    """Should calculate weighted average correctly."""
    scores = {
        EvaluationDimension.RELEVANCE: 0.8,
        EvaluationDimension.CONSTRAINT_SATISFACTION: 0.6,
        EvaluationDimension.ACCURACY: 0.9,
    }

    # Using default weights (0.5, 0.3, 0.2)
    expected = 0.8 * 0.5 + 0.6 * 0.3 + 0.9 * 0.2
    result = calculate_weighted_score(scores, "some_category")

    assert abs(result - expected) < 0.01


def test_get_active_dimensions():
    """Should return dimensions for a category."""
    dims = get_active_dimensions(AdversarialCategory.NEGATION)
    assert len(dims) > 0
    assert EvaluationDimension.NEGATION_RESPECT in dims
