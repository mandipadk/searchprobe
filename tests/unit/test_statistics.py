"""Tests for statistical functions."""

import pytest

from searchprobe.evaluation.statistics import (
    calculate_confidence_interval,
    compare_providers,
    cohens_d,
    failure_mode_frequency,
)


def test_confidence_interval_empty():
    """Empty list should return zero CI."""
    ci = calculate_confidence_interval([])
    assert ci.mean == 0.0
    assert ci.n == 0


def test_confidence_interval_single():
    """Single value should return that value as mean."""
    ci = calculate_confidence_interval([0.75])
    assert ci.mean == 0.75
    assert ci.n == 1


def test_confidence_interval_multiple():
    """Multiple values should calculate CI correctly."""
    scores = [0.5, 0.6, 0.7, 0.8, 0.9]
    ci = calculate_confidence_interval(scores)

    assert ci.mean == 0.7
    assert ci.lower < ci.mean
    assert ci.upper > ci.mean
    assert ci.n == 5


def test_confidence_interval_bounds():
    """CI should be bounded between 0 and 1."""
    # Very tight distribution near 1
    ci = calculate_confidence_interval([0.99, 0.99, 0.99])
    assert ci.upper <= 1.0

    # Very tight distribution near 0
    ci = calculate_confidence_interval([0.01, 0.01, 0.01])
    assert ci.lower >= 0.0


def test_cohens_d_equal():
    """Equal distributions should have d=0."""
    d = cohens_d([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    assert abs(d) < 0.01


def test_cohens_d_different():
    """Different distributions should have non-zero d."""
    d = cohens_d([0.8, 0.9, 0.85], [0.3, 0.4, 0.35])
    assert d > 2.0  # Large effect size


def test_failure_mode_frequency():
    """Should count failure modes correctly."""
    evaluations = [
        {"failure_modes": ["negation_violated", "irrelevant"]},
        {"failure_modes": ["negation_violated"]},
        {"failure_modes": []},
        {"failure_modes": ["timeout"]},
    ]

    counts = failure_mode_frequency(evaluations)

    assert counts["negation_violated"] == 2
    assert counts["irrelevant"] == 1
    assert counts["timeout"] == 1


def test_compare_providers():
    """Should compare two providers statistically."""
    scores_a = [0.8, 0.85, 0.9, 0.75, 0.82]
    scores_b = [0.5, 0.55, 0.6, 0.45, 0.52]

    result = compare_providers("provider_a", scores_a, "provider_b", scores_b)

    assert result.mean_diff > 0  # A is better
    assert result.provider_a == "provider_a"
    assert result.provider_b == "provider_b"
