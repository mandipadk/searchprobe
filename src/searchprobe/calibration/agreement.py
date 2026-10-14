"""Agreement statistics between the LLM judge and human labels."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    """Cohen's kappa for two raters over the same items (nominal labels).

    Returns 1.0 when both raters use a single identical label throughout
    (perfect but uninformative agreement).
    """
    if len(a) != len(b):
        raise ValueError("rater label lists must be the same length")
    n = len(a)
    if n == 0:
        return float("nan")
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[label] * cb[label] for label in set(ca) | set(cb)) / (n * n)
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1 - expected)


def weighted_kappa(
    a: Sequence[int], b: Sequence[int], categories: Sequence[int] = (0, 1, 2, 3)
) -> float:
    """Quadratic-weighted kappa for ordinal labels (e.g. relevance 0-3)."""
    if len(a) != len(b):
        raise ValueError("rater label lists must be the same length")
    n = len(a)
    if n == 0:
        return float("nan")
    k = len(categories)
    idx = {c: i for i, c in enumerate(categories)}
    observed = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b):
        observed[idx[x]][idx[y]] += 1
    row = [sum(r) for r in observed]
    col = [sum(observed[i][j] for i in range(k)) for j in range(k)]
    num = den = 0.0
    for i in range(k):
        for j in range(k):
            w = ((i - j) ** 2) / ((k - 1) ** 2)
            num += w * observed[i][j]
            den += w * row[i] * col[j] / n
    if den == 0:
        return 1.0
    return 1 - num / den


def confusion(a: Sequence[str], b: Sequence[str]) -> dict[tuple[str, str], int]:
    """Counts of (rater_a, rater_b) label pairs."""
    return dict(Counter(zip(a, b)))


def interpret_kappa(kappa: float) -> str:
    """Landis & Koch (1977) qualitative bands."""
    if kappa != kappa:  # NaN
        return "n/a"
    if kappa < 0:
        return "poor"
    for threshold, label in (
        (0.2, "slight"),
        (0.4, "fair"),
        (0.6, "moderate"),
        (0.8, "substantial"),
    ):
        if kappa <= threshold:
            return label
    return "almost perfect"
