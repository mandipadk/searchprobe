"""Operator Blindness Index and invariance sensitivity.

For a system S, base query q, and variant v, let

    O(q, v)     = RBO_ext(results_S(q), results_S(v))
    noise(q)    = O(q, q)  measured by re-issuing q (S's own run-to-run variation)
    unrelated   = mean O between results for *different* base cases (floor)

Then for a meaning-changing (DIR) variant:

    OBI = (O(q, v) - unrelated) / (noise(q) - unrelated)

OBI ~ 1: the engine returns what it would return for the same query again; it
ignores the operator entirely. OBI ~ 0: results differ as much as for an
unrelated query. Values are clipped to [0, 1.5] (above 1 means more similar
than the engine's own noise floor).

For a meaning-preserving (INV) variant we report ``drift = 1 - normalized
overlap``: 0 is perfectly invariant, 1 is as different as an unrelated query.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass

from searchprobe.evaluation.statistics import ConfidenceInterval, bootstrap_confidence_interval
from searchprobe.perturbation.stability import rank_biased_overlap

OBI_CLIP = 1.5


@dataclass
class VariantResult:
    system: str
    case_id: str
    category: str
    kind: str  # DIR | INV | REPEAT
    relation: str
    text: str
    base_urls: list[str]
    variant_urls: list[str]
    error: str | None = None

    @property
    def overlap(self) -> float:
        return rank_biased_overlap(self.base_urls, self.variant_urls)


@dataclass
class OBIRow:
    system: str
    case_id: str
    category: str
    relation: str
    overlap: float
    noise: float
    unrelated: float
    obi: float


@dataclass
class OBISummary:
    system: str
    group: str  # relation or category
    obi: ConfidenceInterval
    n: int


def unrelated_floor(base_lists: dict[str, list[str]], max_pairs: int = 500) -> float:
    """Mean overlap between result lists of different base cases."""
    keys = sorted(base_lists)
    pairs = list(itertools.combinations(keys, 2))[:max_pairs]
    if not pairs:
        return 0.0
    return sum(rank_biased_overlap(base_lists[a], base_lists[b]) for a, b in pairs) / len(pairs)


def normalized(overlap: float, noise: float, floor: float) -> float:
    denom = noise - floor
    if denom <= 1e-9:
        return float("nan")  # engine is as noisy as unrelated queries: undefined
    return max(0.0, min(OBI_CLIP, (overlap - floor) / denom))


def compute_obi(results: list[VariantResult]) -> tuple[list[OBIRow], list[OBIRow]]:
    """Per-case OBI rows for DIR variants and drift rows for INV variants."""
    ok = [r for r in results if r.error is None]
    by_system: dict[str, list[VariantResult]] = defaultdict(list)
    for r in ok:
        by_system[r.system].append(r)

    dir_rows: list[OBIRow] = []
    inv_rows: list[OBIRow] = []
    for system, rows in by_system.items():
        base_lists = {r.case_id: r.base_urls for r in rows}
        floor = unrelated_floor(base_lists)
        noise_by_case = defaultdict(list)
        for r in rows:
            if r.kind == "REPEAT":
                noise_by_case[r.case_id].append(r.overlap)
        system_noise = (
            sum(v for vs in noise_by_case.values() for v in vs)
            / max(1, sum(len(vs) for vs in noise_by_case.values()))
            if noise_by_case
            else 1.0
        )
        for r in rows:
            if r.kind == "REPEAT":
                continue
            noise = (
                sum(noise_by_case[r.case_id]) / len(noise_by_case[r.case_id])
                if noise_by_case.get(r.case_id)
                else system_noise
            )
            value = normalized(r.overlap, noise, floor)
            if value != value:  # NaN
                continue
            row = OBIRow(
                system,
                r.case_id,
                r.category,
                r.relation,
                r.overlap,
                noise,
                floor,
                value if r.kind == "DIR" else 1.0 - min(1.0, value),
            )
            (dir_rows if r.kind == "DIR" else inv_rows).append(row)
    return dir_rows, inv_rows


def summarize(rows: list[OBIRow], by: str = "relation") -> list[OBISummary]:
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        groups[(r.system, getattr(r, by))].append(r.obi)
        groups[(r.system, "ALL")].append(r.obi)
    return [
        OBISummary(system, group, bootstrap_confidence_interval(vals, n_bootstrap=2000), len(vals))
        for (system, group), vals in sorted(groups.items())
    ]
