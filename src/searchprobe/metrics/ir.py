"""Rank-aware retrieval metrics over per-document judgments.

Headline metric for the adversarial categories is ``violation@k``: the share
of the top-k results that *break* a hard constraint of the query (an AI
company returned for "companies not in AI"). It is computed over documents
that were judged; unjudged positions are reported as ``coverage``.

nDCG uses graded relevance (0-3) zeroed for violating documents, with the
ideal ranking taken from the pool of all judged documents for the case
across every system (TREC-style pooling), so systems are compared against
the same ideal.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from searchprobe.cases.models import TestCase
from searchprobe.judging.models import DocJudgment


@dataclass
class ListMetrics:
    """Metrics for one system's ranked list on one case."""

    case_id: str
    system: str
    category: str
    k: int
    n_returned: int
    n_judged: int
    coverage: float  # judged / min(k, returned)
    violation_at_k: float | None  # share of judged top-k that violate a hard constraint
    satisfied_at_k: float | None  # share of judged top-k satisfying every hard constraint
    precision_at_k: float | None  # share of judged top-k with relevance >= 2 and no violation
    ndcg_at_k: float | None
    reciprocal_rank: float  # 1 / rank of first fully satisfying doc (0 if none)
    any_violation: bool
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def dcg(gains: list[float]) -> float:
    return sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(gains))


def list_metrics(
    case: TestCase,
    system: str,
    judgments: list[DocJudgment | None],
    pool: list[DocJudgment],
    k: int = 10,
    error: str | None = None,
) -> ListMetrics:
    """Compute metrics for one ranked list.

    Args:
        case: The test case.
        system: System key.
        judgments: Judgment per rank position (None if unjudged).
        pool: All judgments for this case across systems (for the ideal DCG).
        k: Cutoff.
        error: Search error, if the system failed (metrics are then None).
    """
    top = judgments[:k]
    judged = [j for j in top if j is not None]
    n_returned = len(judgments)
    base = dict(case_id=case.id, system=system, category=case.category.value, k=k)

    if error:
        return ListMetrics(
            **base,
            n_returned=0,
            n_judged=0,
            coverage=0.0,
            violation_at_k=None,
            satisfied_at_k=None,
            precision_at_k=None,
            ndcg_at_k=None,
            reciprocal_rank=0.0,
            any_violation=False,
            error=error,
        )
    if not top:
        # An empty result list is a retrieval failure: nothing satisfying, nothing violating.
        return ListMetrics(
            **base,
            n_returned=0,
            n_judged=0,
            coverage=1.0,
            violation_at_k=0.0,
            satisfied_at_k=0.0,
            precision_at_k=0.0,
            ndcg_at_k=0.0,
            reciprocal_rank=0.0,
            any_violation=False,
        )

    coverage = len(judged) / len(top)
    if judged:
        violating = [j.is_violating(case) for j in judged]
        satisfying = [j.is_fully_satisfying(case) for j in judged]
        relevant = [j.relevance >= 2 and not v for j, v in zip(judged, violating)]
        violation = sum(violating) / len(judged)
        satisfied = sum(satisfying) / len(judged)
        precision = sum(relevant) / len(judged)
    else:
        violating, violation = [], None
        satisfied = precision = None

    rr = 0.0
    for rank, j in enumerate(top, 1):
        if j is not None and j.is_fully_satisfying(case):
            rr = 1.0 / rank
            break

    ndcg = None
    if judged:
        gains = [j.gain(case) if j is not None else 0.0 for j in top]
        # Unique documents in the pool, best gains first.
        best_by_doc: dict[str, float] = {}
        for p in [*pool, *judged]:
            best_by_doc[p.content_hash] = max(best_by_doc.get(p.content_hash, 0.0), p.gain(case))
        ideal = sorted(best_by_doc.values(), reverse=True)[:k]
        idcg = dcg(ideal)
        ndcg = dcg(gains) / idcg if idcg > 0 else 0.0

    return ListMetrics(
        **base,
        n_returned=n_returned,
        n_judged=len(judged),
        coverage=coverage,
        violation_at_k=violation,
        satisfied_at_k=satisfied,
        precision_at_k=precision,
        ndcg_at_k=ndcg,
        reciprocal_rank=rr,
        any_violation=any(violating),
    )
