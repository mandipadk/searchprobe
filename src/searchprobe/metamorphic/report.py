"""Persist and render metamorphic runs."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from searchprobe.metamorphic.engine import MetamorphicRun
from searchprobe.metamorphic.obi import OBIRow, compute_obi, summarize

SCHEMA = """
CREATE TABLE IF NOT EXISTS metamorphic_results (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    system TEXT NOT NULL,
    case_id TEXT NOT NULL,
    category TEXT,
    kind TEXT NOT NULL,
    relation TEXT NOT NULL,
    variant_text TEXT,
    base_urls TEXT,
    variant_urls TEXT,
    overlap REAL,
    error TEXT,
    timestamp TEXT
);
CREATE INDEX IF NOT EXISTS idx_metamorphic_run ON metamorphic_results(run_id);
"""


def save_run(db_path: str | Path, run: MetamorphicRun) -> str:
    run_id = str(uuid4())
    now = datetime.now(UTC).isoformat()
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.executemany(
            """INSERT INTO metamorphic_results
               (id, run_id, system, case_id, category, kind, relation, variant_text, base_urls,
                variant_urls, overlap, error, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    str(uuid4()),
                    run_id,
                    r.system,
                    r.case_id,
                    r.category,
                    r.kind,
                    r.relation,
                    r.text,
                    json.dumps(r.base_urls),
                    json.dumps(r.variant_urls),
                    r.overlap if r.error is None else None,
                    r.error,
                    now,
                )
                for r in run.results
            ],
        )
        conn.commit()
    finally:
        conn.close()
    return run_id


def _table(rows: list[OBIRow], by: str, title: str, label: str) -> list[str]:
    summaries = summarize(rows, by=by)
    systems = sorted({s.system for s in summaries})
    groups = sorted({s.group for s in summaries}, key=lambda g: (g == "ALL", g))
    cell = {(s.system, s.group): s for s in summaries}
    lines = [
        f"### {title}",
        "",
        f"| {label} | " + " | ".join(systems) + " |",
        "|---" * (len(systems) + 1) + "|",
    ]
    for g in groups:
        vals = []
        for s in systems:
            c = cell.get((s, g))
            vals.append(
                f"{c.obi.mean:.2f} [{c.obi.lower:.2f}, {c.obi.upper:.2f}] (n={c.n})" if c else "-"
            )
        lines.append(f"| {'**all**' if g == 'ALL' else g} | " + " | ".join(vals) + " |")
    return lines + [""]


def render_markdown(run: MetamorphicRun) -> str:
    dir_rows, inv_rows = compute_obi(run.results)
    lines = [
        "# Metamorphic Test Report",
        "",
        "**Operator Blindness Index (OBI)**: how similar results are for a query and its "
        "meaning-changing twin, normalized between the engine's own run-to-run noise (1.0) and "
        "unrelated queries (0.0). Lower is better: 1.0 means the operator was ignored.",
        "",
        f"Searches: {run.searches}, cost ${run.cost_usd:.4f}, skipped for budget: "
        f"{run.skipped_for_budget}.",
        "",
    ]
    if dir_rows:
        lines += _table(dir_rows, "relation", "OBI by operator (↓ better)", "Relation")
        lines += _table(dir_rows, "category", "OBI by category (↓ better)", "Category")
        noise = {}
        for r in dir_rows:
            noise.setdefault(r.system, (r.noise, r.unrelated))
        lines += [
            "### Baselines",
            "",
            "| System | Noise floor RBO (repeat) | Unrelated RBO |",
            "|---|---|---|",
        ]
        lines += [f"| {s} | {n:.2f} | {u:.2f} |" for s, (n, u) in sorted(noise.items())]
        lines.append("")
        worst = sorted(dir_rows, key=lambda r: r.obi, reverse=True)[:10]
        lines += ["### Most operator-blind cases", ""]
        lines += [
            f"- {r.system} · {r.category} · {r.relation}: OBI {r.obi:.2f} "
            f"(overlap {r.overlap:.2f})  `{r.case_id}`"
            for r in worst
        ]
        lines.append("")
    if inv_rows:
        lines += _table(
            inv_rows, "relation", "Drift under meaning-preserving edits (↓ better)", "Relation"
        )
    if run.directions:
        lines += [
            "### Judged direction (DIR)",
            "",
            "Share of top results satisfying the twin's constraints: variant results "
            "minus base results (↑ better: the engine moved toward the right answers).",
            "",
            "| System | Mean direction | n |",
            "|---|---|---|",
        ]
        by_sys: dict[str, list[float]] = {}
        for d in run.directions:
            by_sys.setdefault(d.system, []).append(d.direction)
        lines += [f"| {s} | {sum(v) / len(v):+.2f} | {len(v)} |" for s, v in sorted(by_sys.items())]
        lines.append("")
    return "\n".join(lines)
