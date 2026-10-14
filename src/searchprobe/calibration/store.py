"""Human labels for judge calibration, and judge-vs-human agreement reports."""

from __future__ import annotations

import random
import sqlite3
from collections import defaultdict
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from searchprobe.calibration.agreement import cohen_kappa, interpret_kappa, weighted_kappa
from searchprobe.cases.models import TestCase
from searchprobe.judging.models import DocJudgment

RELEVANCE_KEY = "__relevance__"


@dataclass
class LabelItem:
    """One (case, document) pair queued for human labeling."""

    case: TestCase
    judgment: DocJudgment


class HumanLabelStore:
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS human_labels (
        case_id TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        constraint_id TEXT NOT NULL,   -- or '__relevance__'
        label TEXT NOT NULL,           -- verdict, or relevance as '0'..'3'
        labeler TEXT NOT NULL,
        note TEXT,
        created_at TEXT NOT NULL,
        PRIMARY KEY (case_id, content_hash, constraint_id, labeler)
    );
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        with self._conn() as conn:
            conn.executescript(self.SCHEMA)
            conn.commit()

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def put(
        self,
        case_id: str,
        content_hash: str,
        constraint_id: str,
        label: str,
        labeler: str = "human",
        note: str | None = None,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO human_labels
                   (case_id, content_hash, constraint_id, label, labeler, note, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    case_id,
                    content_hash,
                    constraint_id,
                    label,
                    labeler,
                    note,
                    datetime.now(UTC).isoformat(),
                ),
            )
            conn.commit()

    def labeled_pairs(self, labeler: str | None = None) -> set[tuple[str, str]]:
        query = "SELECT DISTINCT case_id, content_hash FROM human_labels"
        params: list[str] = []
        if labeler:
            query += " WHERE labeler = ?"
            params.append(labeler)
        with self._conn() as conn:
            return {(r[0], r[1]) for r in conn.execute(query, params)}

    def all(self, labeler: str | None = None) -> dict[tuple[str, str, str], str]:
        query = "SELECT case_id, content_hash, constraint_id, label FROM human_labels"
        params: list[str] = []
        if labeler:
            query += " WHERE labeler = ?"
            params.append(labeler)
        with self._conn() as conn:
            return {(r[0], r[1], r[2]): r[3] for r in conn.execute(query, params)}


def stratified_sample(
    judgments: list[DocJudgment],
    cases: dict[str, TestCase],
    n: int = 200,
    seed: int = 0,
    exclude: set[tuple[str, str]] | None = None,
) -> list[LabelItem]:
    """Sample (case, doc) pairs balanced across category and judge verdict.

    Balancing on the judge's own verdict (violating / satisfying / other)
    ensures the rare outcomes that matter most get enough labels.
    """
    exclude = exclude or set()
    strata: dict[tuple[str, str], list[LabelItem]] = defaultdict(list)
    for j in judgments:
        case = cases.get(j.case_id)
        if case is None or (j.case_id, j.content_hash) in exclude:
            continue
        outcome = (
            "violating"
            if j.is_violating(case)
            else "satisfying"
            if j.is_fully_satisfying(case)
            else "other"
        )
        strata[(case.category.value, outcome)].append(LabelItem(case, j))
    rng = random.Random(seed)
    for items in strata.values():
        rng.shuffle(items)
    sample: list[LabelItem] = []
    keys = sorted(strata)
    while len(sample) < n and any(strata[k] for k in keys):
        for k in keys:
            if strata[k] and len(sample) < n:
                sample.append(strata[k].pop())
    return sample


@dataclass
class AgreementReport:
    n_pairs: int
    relevance_kappa: float | None
    verdict_kappa_by_type: dict[str, tuple[float, int]] = field(default_factory=dict)
    overall_verdict_kappa: float | None = None
    accuracy_by_type: dict[str, float] = field(default_factory=dict)

    def lines(self) -> list[str]:
        out = [f"Labeled pairs: {self.n_pairs}"]
        if self.relevance_kappa is not None:
            out.append(
                f"Relevance (quadratic-weighted kappa): {self.relevance_kappa:.2f} "
                f"({interpret_kappa(self.relevance_kappa)})"
            )
        if self.overall_verdict_kappa is not None:
            out.append(
                f"Constraint verdicts (Cohen's kappa): {self.overall_verdict_kappa:.2f} "
                f"({interpret_kappa(self.overall_verdict_kappa)})"
            )
        for ctype, (kappa, n) in sorted(self.verdict_kappa_by_type.items()):
            acc = self.accuracy_by_type.get(ctype, float("nan"))
            out.append(f"  {ctype:<10} kappa={kappa:.2f} accuracy={acc:.0%} n={n}")
        return out


def agreement_report(
    labels: dict[tuple[str, str, str], str],
    judgments: list[DocJudgment],
    cases: dict[str, TestCase],
) -> AgreementReport:
    """Compare stored human labels with judge outputs."""
    by_pair = {(j.case_id, j.content_hash): j for j in judgments}
    rel_h: list[int] = []
    rel_j: list[int] = []
    verdicts: dict[str, tuple[list[str], list[str]]] = defaultdict(lambda: ([], []))
    pairs = set()
    for (case_id, chash, cid), label in labels.items():
        j = by_pair.get((case_id, chash))
        case = cases.get(case_id)
        if j is None or case is None:
            continue
        pairs.add((case_id, chash))
        if cid == RELEVANCE_KEY:
            rel_h.append(int(label))
            rel_j.append(j.relevance)
            continue
        try:
            ctype = case.constraint(cid).type
        except KeyError:
            continue
        h, m = verdicts[ctype]
        h.append(label)
        m.append(j.verdict_for(cid))

    all_h = [x for h, _ in verdicts.values() for x in h]
    all_m = [x for _, m in verdicts.values() for x in m]
    report = AgreementReport(
        n_pairs=len(pairs),
        relevance_kappa=weighted_kappa(rel_h, rel_j) if rel_h else None,
        overall_verdict_kappa=cohen_kappa(all_h, all_m) if all_h else None,
    )
    for ctype, (h, m) in verdicts.items():
        report.verdict_kappa_by_type[ctype] = (cohen_kappa(h, m), len(h))
        report.accuracy_by_type[ctype] = sum(x == y for x, y in zip(h, m)) / len(h)
    return report
