"""Statistical analysis for evaluation results."""

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats


@dataclass
class ConfidenceInterval:
    """Confidence interval for a metric."""

    mean: float
    lower: float
    upper: float
    confidence: float = 0.95
    n: int = 0

    def __str__(self) -> str:
        return f"{self.mean:.3f} [{self.lower:.3f}, {self.upper:.3f}]"


@dataclass
class ComparisonResult:
    """Result of comparing two providers/modes."""

    provider_a: str
    provider_b: str
    mean_diff: float  # A - B
    ci: ConfidenceInterval
    p_value: float
    significant: bool  # at alpha=0.05
    effect_size: float  # Cohen's d


def calculate_confidence_interval(
    scores: list[float],
    confidence: float = 0.95,
) -> ConfidenceInterval:
    """Calculate confidence interval for mean score.

    Args:
        scores: List of scores (0-1)
        confidence: Confidence level (default 0.95)

    Returns:
        ConfidenceInterval with mean, lower, upper bounds
    """
    if not scores:
        return ConfidenceInterval(
            mean=0.0, lower=0.0, upper=0.0, confidence=confidence, n=0
        )

    n = len(scores)
    mean = np.mean(scores)

    if n == 1:
        return ConfidenceInterval(
            mean=float(mean),
            lower=float(mean),
            upper=float(mean),
            confidence=confidence,
            n=n,
        )

    # Use t-distribution for small samples
    std_err = stats.sem(scores)
    alpha = 1 - confidence
    t_value = stats.t.ppf(1 - alpha / 2, n - 1)

    margin = t_value * std_err
    lower = max(0.0, mean - margin)
    upper = min(1.0, mean + margin)

    return ConfidenceInterval(
        mean=float(mean),
        lower=float(lower),
        upper=float(upper),
        confidence=confidence,
        n=n,
    )


def wilcoxon_signed_rank_test(
    scores_a: list[float],
    scores_b: list[float],
) -> tuple[float, float]:
    """Perform Wilcoxon signed-rank test for paired samples.

    Args:
        scores_a: Scores from provider/mode A
        scores_b: Scores from provider/mode B (paired with A)

    Returns:
        Tuple of (statistic, p_value)
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("Score lists must have same length for paired test")

    if len(scores_a) < 5:
        # Not enough samples for meaningful test
        return 0.0, 1.0

    # Filter out ties (where difference is 0)
    diffs = [a - b for a, b in zip(scores_a, scores_b)]
    non_zero_diffs = [d for d in diffs if d != 0]

    if len(non_zero_diffs) < 5:
        return 0.0, 1.0

    try:
        statistic, p_value = stats.wilcoxon(scores_a, scores_b)
        return float(statistic), float(p_value)
    except Exception:
        return 0.0, 1.0


def mann_whitney_test(
    scores_a: list[float],
    scores_b: list[float],
) -> tuple[float, float]:
    """Perform Mann-Whitney U test for independent samples.

    Args:
        scores_a: Scores from provider/mode A
        scores_b: Scores from provider/mode B

    Returns:
        Tuple of (statistic, p_value)
    """
    if len(scores_a) < 3 or len(scores_b) < 3:
        return 0.0, 1.0

    try:
        statistic, p_value = stats.mannwhitneyu(
            scores_a, scores_b, alternative="two-sided"
        )
        return float(statistic), float(p_value)
    except Exception:
        return 0.0, 1.0


def cohens_d(scores_a: list[float], scores_b: list[float]) -> float:
    """Calculate Cohen's d effect size.

    Args:
        scores_a: Scores from provider/mode A
        scores_b: Scores from provider/mode B

    Returns:
        Cohen's d (standardized mean difference)
    """
    if not scores_a or not scores_b:
        return 0.0

    mean_a = np.mean(scores_a)
    mean_b = np.mean(scores_b)

    # Pooled standard deviation
    n_a = len(scores_a)
    n_b = len(scores_b)

    var_a = np.var(scores_a, ddof=1) if n_a > 1 else 0
    var_b = np.var(scores_b, ddof=1) if n_b > 1 else 0

    pooled_std = math.sqrt(
        ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    )

    if pooled_std == 0:
        return 0.0

    return float((mean_a - mean_b) / pooled_std)


def bootstrap_confidence_interval(
    scores: list[float],
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    statistic: str = "mean",
) -> ConfidenceInterval:
    """Calculate bootstrap confidence interval.

    Uses resampling to estimate CI without parametric assumptions.

    Args:
        scores: List of scores
        confidence: Confidence level (default 0.95)
        n_bootstrap: Number of bootstrap samples
        statistic: Statistic to compute ('mean' or 'median')

    Returns:
        ConfidenceInterval from bootstrap resampling
    """
    if not scores:
        return ConfidenceInterval(mean=0.0, lower=0.0, upper=0.0, confidence=confidence, n=0)

    n = len(scores)
    arr = np.array(scores)
    stat_func = np.mean if statistic == "mean" else np.median
    observed = float(stat_func(arr))

    if n == 1:
        return ConfidenceInterval(
            mean=observed, lower=observed, upper=observed, confidence=confidence, n=1
        )

    rng = np.random.default_rng(42)
    bootstrap_stats = np.array([
        float(stat_func(rng.choice(arr, size=n, replace=True)))
        for _ in range(n_bootstrap)
    ])

    alpha = 1 - confidence
    lower = float(np.percentile(bootstrap_stats, 100 * alpha / 2))
    upper = float(np.percentile(bootstrap_stats, 100 * (1 - alpha / 2)))

    return ConfidenceInterval(
        mean=observed, lower=lower, upper=upper, confidence=confidence, n=n
    )


def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Apply Benjamini-Hochberg procedure for multiple testing correction.

    Controls the False Discovery Rate (FDR) at level alpha.

    Args:
        p_values: List of p-values from multiple tests
        alpha: Desired FDR level (default 0.05)

    Returns:
        List of booleans indicating which tests are significant after correction
    """
    if not p_values:
        return []

    m = len(p_values)
    # Sort p-values and track original indices
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])

    significant = [False] * m
    # Find the largest k such that p_(k) <= k/m * alpha
    max_k = -1
    for k, (orig_idx, p_val) in enumerate(indexed, 1):
        threshold = (k / m) * alpha
        if p_val <= threshold:
            max_k = k

    # All tests with rank <= max_k are significant
    if max_k > 0:
        for k in range(max_k):
            orig_idx = indexed[k][0]
            significant[orig_idx] = True

    return significant


def compare_providers(
    provider_a: str,
    scores_a: list[float],
    provider_b: str,
    scores_b: list[float],
    paired: bool = True,
) -> ComparisonResult:
    """Compare two providers statistically.

    Args:
        provider_a: Name of first provider
        scores_a: Scores from provider A
        provider_b: Name of second provider
        scores_b: Scores from provider B
        paired: Whether samples are paired (same queries)

    Returns:
        ComparisonResult with statistical analysis
    """
    mean_a = np.mean(scores_a) if scores_a else 0.0
    mean_b = np.mean(scores_b) if scores_b else 0.0
    mean_diff = float(mean_a - mean_b)

    # Confidence interval for difference
    if paired and len(scores_a) == len(scores_b):
        diffs = [a - b for a, b in zip(scores_a, scores_b)]
        diff_ci = calculate_confidence_interval(diffs)
        _, p_value = wilcoxon_signed_rank_test(scores_a, scores_b)
    else:
        diff_ci = ConfidenceInterval(
            mean=mean_diff,
            lower=mean_diff - 0.1,  # Approximation
            upper=mean_diff + 0.1,
            n=min(len(scores_a), len(scores_b)),
        )
        _, p_value = mann_whitney_test(scores_a, scores_b)

    effect = cohens_d(scores_a, scores_b)

    return ComparisonResult(
        provider_a=provider_a,
        provider_b=provider_b,
        mean_diff=mean_diff,
        ci=diff_ci,
        p_value=p_value,
        significant=p_value < 0.05,
        effect_size=effect,
    )


def aggregate_by_category(
    evaluations: list[dict],
) -> dict[str, dict[str, ConfidenceInterval]]:
    """Aggregate scores by category and provider.

    Args:
        evaluations: List of evaluation dicts with category, provider, weighted_score

    Returns:
        Nested dict: category -> provider -> ConfidenceInterval
    """
    # Group scores
    grouped: dict[str, dict[str, list[float]]] = {}

    for eval_item in evaluations:
        category = eval_item.get("category", "unknown")
        provider_key = f"{eval_item.get('provider', 'unknown')}:{eval_item.get('search_mode', 'default')}"
        score = eval_item.get("weighted_score", 0.0)

        if category not in grouped:
            grouped[category] = {}
        if provider_key not in grouped[category]:
            grouped[category][provider_key] = []

        grouped[category][provider_key].append(score)

    # Calculate CIs
    result: dict[str, dict[str, ConfidenceInterval]] = {}

    for category, providers in grouped.items():
        result[category] = {}
        for provider, scores in providers.items():
            result[category][provider] = calculate_confidence_interval(scores)

    return result


def aggregate_by_provider(
    evaluations: list[dict],
) -> dict[str, ConfidenceInterval]:
    """Aggregate scores by provider across all categories.

    Args:
        evaluations: List of evaluation dicts

    Returns:
        Dict: provider -> ConfidenceInterval
    """
    grouped: dict[str, list[float]] = {}

    for eval_item in evaluations:
        provider_key = f"{eval_item.get('provider', 'unknown')}:{eval_item.get('search_mode', 'default')}"
        score = eval_item.get("weighted_score", 0.0)

        if provider_key not in grouped:
            grouped[provider_key] = []
        grouped[provider_key].append(score)

    return {
        provider: calculate_confidence_interval(scores)
        for provider, scores in grouped.items()
    }


def failure_mode_frequency(
    evaluations: list[dict],
) -> dict[str, int]:
    """Count frequency of each failure mode.

    Args:
        evaluations: List of evaluation dicts

    Returns:
        Dict: failure_mode -> count
    """
    counts: dict[str, int] = {}

    for eval_item in evaluations:
        for mode in eval_item.get("failure_modes", []):
            counts[mode] = counts.get(mode, 0) + 1

    # Sort by frequency
    return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))


def summary_statistics(evaluations: list[dict]) -> dict[str, Any]:
    """Calculate summary statistics for evaluation results.

    Args:
        evaluations: List of evaluation dicts

    Returns:
        Dictionary with summary statistics
    """
    if not evaluations:
        return {"count": 0}

    scores = [e.get("weighted_score", 0.0) for e in evaluations]

    return {
        "count": len(evaluations),
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "ci_95": calculate_confidence_interval(scores),
        "by_provider": aggregate_by_provider(evaluations),
        "by_category": aggregate_by_category(evaluations),
        "failure_modes": failure_mode_frequency(evaluations),
    }
