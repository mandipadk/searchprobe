"""Triplet probes and per-case failure prediction.

Probe: for each scorer, pairwise accuracy P[s(q, d+) > s(q, d-)] over
triplets (0.5 = chance), with the margin distribution, overall and per
category, with bootstrap CIs.

Prediction: does a case's triplet failure rate on open models predict whether
a search provider *violated* that case (from the per-document judgments)?
Measured as ROC AUC with a bootstrap CI over cases (n = hundreds of cases,
not 13 category means).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from searchprobe.evaluation.statistics import ConfidenceInterval, bootstrap_confidence_interval
from searchprobe.explain.scorers import Scorer
from searchprobe.explain.triplets import Triplet


@dataclass
class TripletScore:
    case_id: str
    category: str
    scorer: str
    positive_score: float
    negative_score: float

    @property
    def correct(self) -> bool:
        return self.positive_score > self.negative_score

    @property
    def margin(self) -> float:
        return self.positive_score - self.negative_score


@dataclass
class ProbeSummary:
    scorer: str
    group: str
    accuracy: ConfidenceInterval
    mean_margin: float
    n: int


def run_probe(triplets: list[Triplet], scorers: list[Scorer]) -> list[TripletScore]:
    out: list[TripletScore] = []
    for scorer in scorers:
        # Score per query so corpus-dependent scorers (BM25 IDF) see the case's documents.
        by_query: dict[str, list[Triplet]] = defaultdict(list)
        for t in triplets:
            by_query[t.query].append(t)
        for query, ts in by_query.items():
            docs = [d for t in ts for d in (t.positive, t.negative)]
            scores = scorer.score(query, docs)
            for i, t in enumerate(ts):
                out.append(
                    TripletScore(
                        t.case_id,
                        t.category,
                        scorer.name,
                        float(scores[2 * i]),
                        float(scores[2 * i + 1]),
                    )
                )
    return out


def summarize_probe(scores: list[TripletScore], by: str = "category") -> list[ProbeSummary]:
    groups: dict[tuple[str, str], list[TripletScore]] = defaultdict(list)
    for s in scores:
        groups[(s.scorer, getattr(s, by))].append(s)
        groups[(s.scorer, "ALL")].append(s)
    return [
        ProbeSummary(
            scorer,
            group,
            bootstrap_confidence_interval([float(x.correct) for x in xs], n_bootstrap=2000),
            float(np.mean([x.margin for x in xs])),
            len(xs),
        )
        for (scorer, group), xs in sorted(groups.items())
    ]


def roc_auc(scores: list[float], labels: list[int]) -> float:
    """ROC AUC via the Mann-Whitney statistic (ties count half)."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


@dataclass
class PredictionResult:
    scorer: str
    system: str
    auc: float
    lower: float
    upper: float
    n_cases: int
    n_violated: int


def predict_failures(
    scores: list[TripletScore],
    violated: dict[tuple[str, str], bool],
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> list[PredictionResult]:
    """AUC of 'triplet failure rate' predicting provider violation, per (scorer, system).

    Args:
        scores: Triplet scores.
        violated: (system, case_id) -> whether the system returned a violating
            document in its top-k for that case.
    """
    fail_rate: dict[tuple[str, str], list[float]] = defaultdict(list)
    for s in scores:
        fail_rate[(s.scorer, s.case_id)].append(0.0 if s.correct else 1.0)
    systems = sorted({sys for sys, _ in violated})
    scorers = sorted({sc for sc, _ in fail_rate})
    rng = np.random.default_rng(seed)
    out = []
    for scorer in scorers:
        for system in systems:
            xs, ys = [], []
            for (sc, cid), rates in fail_rate.items():
                if sc == scorer and (system, cid) in violated:
                    xs.append(float(np.mean(rates)))
                    ys.append(int(violated[(system, cid)]))
            if len(set(ys)) < 2:
                continue
            auc = roc_auc(xs, ys)
            boots = []
            idx = np.arange(len(xs))
            for _ in range(n_bootstrap):
                b = rng.choice(idx, size=len(idx), replace=True)
                bx, by = [xs[i] for i in b], [ys[i] for i in b]
                if len(set(by)) == 2:
                    boots.append(roc_auc(bx, by))
            lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (auc, auc)
            out.append(
                PredictionResult(scorer, system, auc, float(lo), float(hi), len(xs), sum(ys))
            )
    return out


def probe_markdown(
    scores: list[TripletScore], predictions: list[PredictionResult] | None = None
) -> str:
    lines = [
        "# Triplet Probe Report",
        "",
        "Pairwise accuracy: how often a scorer ranks the document that satisfies the query "
        "above one that violates it (0.5 = chance). Brackets: 95% bootstrap CIs.",
        "",
    ]
    summaries = summarize_probe(scores)
    scorers = sorted({s.scorer for s in summaries})
    groups = sorted({s.group for s in summaries}, key=lambda g: (g == "ALL", g))
    cell = {(s.scorer, s.group): s for s in summaries}
    lines += ["| Category | " + " | ".join(scorers) + " |", "|---" * (len(scorers) + 1) + "|"]
    for g in groups:
        row = []
        for sc in scorers:
            c = cell.get((sc, g))
            row.append(
                f"{c.accuracy.mean:.0%} [{c.accuracy.lower:.0%}, {c.accuracy.upper:.0%}] (n={c.n})"
                if c
                else "-"
            )
        lines.append(f"| {'**all**' if g == 'ALL' else g} | " + " | ".join(row) + " |")
    lines.append("")
    if predictions:
        lines += [
            "## Do triplet failures predict provider failures?",
            "",
            "ROC AUC of a case's triplet failure rate predicting that the provider returned a "
            "violating document for the case (0.5 = no relationship).",
            "",
            "| Scorer | System | AUC [95% CI] | Cases | Violated |",
            "|---|---|---|---|---|",
        ]
        for p in predictions:
            lines.append(
                f"| {p.scorer} | {p.system} | {p.auc:.2f} [{p.lower:.2f}, {p.upper:.2f}] "
                f"| {p.n_cases} | {p.n_violated} |"
            )
        lines.append("")
    return "\n".join(lines)
