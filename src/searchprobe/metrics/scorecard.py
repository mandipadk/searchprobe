"""Aggregate per-list metrics into a scorecard with uncertainty.

Every number in a scorecard comes with a bootstrap confidence interval over
cases and a sample size; system-vs-system differences are paired by case and
corrected for multiple comparisons (Benjamini-Hochberg).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from searchprobe.evaluation.statistics import (
    ConfidenceInterval,
    benjamini_hochberg,
    bootstrap_confidence_interval,
    wilcoxon_signed_rank_test,
)
from searchprobe.judging.engine import RunJudgments
from searchprobe.metrics.ir import ListMetrics, list_metrics

METRICS = ("violation_at_k", "satisfied_at_k", "precision_at_k", "ndcg_at_k", "reciprocal_rank")
LOWER_IS_BETTER = {"violation_at_k"}


def compute_list_metrics(run: RunJudgments, k: int = 10) -> list[ListMetrics]:
    """Metrics for every (case, system) list, with pooled ideal rankings per case."""
    pools: dict[str, list] = defaultdict(list)
    for jl in run.lists:
        pools[jl.case.id].extend(j for j in jl.judgments[:k] if j is not None)
    return [
        list_metrics(jl.case, jl.system, jl.judgments, pools[jl.case.id], k=k, error=jl.error)
        for jl in run.lists
    ]


@dataclass
class Cell:
    """One aggregated number with its CI."""

    ci: ConfidenceInterval

    @property
    def mean(self) -> float:
        return self.ci.mean

    def fmt(self, pct: bool = False) -> str:
        if self.ci.n == 0:
            return "-"
        if pct:
            return f"{self.ci.mean:.0%} [{self.ci.lower:.0%}, {self.ci.upper:.0%}]"
        return f"{self.ci.mean:.3f} [{self.ci.lower:.3f}, {self.ci.upper:.3f}]"


@dataclass
class PairedComparison:
    metric: str
    system_a: str
    system_b: str
    diff: ConfidenceInterval  # mean(A - B) over shared cases
    p_value: float
    significant: bool  # after BH correction
    n: int


@dataclass
class Scorecard:
    metric_rows: list[ListMetrics]
    by_system: dict[str, dict[str, Cell]] = field(default_factory=dict)
    by_category: dict[str, dict[str, dict[str, Cell]]] = field(default_factory=dict)
    comparisons: list[PairedComparison] = field(default_factory=list)
    error_rate: dict[str, float] = field(default_factory=dict)
    coverage: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        def cell(c: Cell) -> dict[str, Any]:
            return {"mean": c.ci.mean, "lower": c.ci.lower, "upper": c.ci.upper, "n": c.ci.n}

        return {
            "by_system": {
                s: {m: cell(c) for m, c in ms.items()} for s, ms in self.by_system.items()
            },
            "by_category": {
                cat: {s: {m: cell(c) for m, c in ms.items()} for s, ms in systems.items()}
                for cat, systems in self.by_category.items()
            },
            "comparisons": [
                {
                    "metric": c.metric,
                    "a": c.system_a,
                    "b": c.system_b,
                    "diff": c.diff.mean,
                    "lower": c.diff.lower,
                    "upper": c.diff.upper,
                    "p": c.p_value,
                    "significant": c.significant,
                    "n": c.n,
                }
                for c in self.comparisons
            ],
            "error_rate": self.error_rate,
            "coverage": self.coverage,
        }


def _values(rows: list[ListMetrics], metric: str) -> list[float]:
    return [getattr(r, metric) for r in rows if getattr(r, metric) is not None]


def build_scorecard(
    rows: list[ListMetrics],
    metrics: tuple[str, ...] = METRICS,
    alpha: float = 0.05,
    n_bootstrap: int = 2000,
) -> Scorecard:
    card = Scorecard(metric_rows=rows)
    by_system: dict[str, list[ListMetrics]] = defaultdict(list)
    by_cat_sys: dict[tuple[str, str], list[ListMetrics]] = defaultdict(list)
    for r in rows:
        by_system[r.system].append(r)
        by_cat_sys[(r.category, r.system)].append(r)

    for system, srows in sorted(by_system.items()):
        ok = [r for r in srows if r.error is None]
        card.error_rate[system] = 1 - len(ok) / len(srows) if srows else 0.0
        card.coverage[system] = sum(r.coverage for r in ok) / len(ok) if ok else 0.0
        card.by_system[system] = {
            m: Cell(bootstrap_confidence_interval(_values(ok, m), n_bootstrap=n_bootstrap))
            for m in metrics
        }
    for (cat, system), crows in sorted(by_cat_sys.items()):
        ok = [r for r in crows if r.error is None]
        card.by_category.setdefault(cat, {})[system] = {
            m: Cell(bootstrap_confidence_interval(_values(ok, m), n_bootstrap=n_bootstrap))
            for m in metrics
        }

    # Paired comparisons on cases both systems answered without error.
    systems = sorted(by_system)
    index = {(r.system, r.case_id): r for r in rows if r.error is None}
    pending: list[PairedComparison] = []
    for metric in metrics:
        for i, a in enumerate(systems):
            for b in systems[i + 1 :]:
                pairs = [
                    (getattr(index[(a, cid)], metric), getattr(index[(b, cid)], metric))
                    for cid in {r.case_id for r in rows}
                    if (a, cid) in index and (b, cid) in index
                ]
                pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
                if len(pairs) < 2:
                    continue
                xs, ys = [p[0] for p in pairs], [p[1] for p in pairs]
                diff = bootstrap_confidence_interval(
                    [x - y for x, y in pairs], n_bootstrap=n_bootstrap
                )
                _, p = wilcoxon_signed_rank_test(xs, ys)
                pending.append(PairedComparison(metric, a, b, diff, p, False, len(pairs)))
    decisions = benjamini_hochberg([c.p_value for c in pending], alpha)
    for comp, sig in zip(pending, decisions):
        comp.significant = sig
    card.comparisons = pending
    return card
