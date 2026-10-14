"""Render scorecards as Markdown (with concrete failure exhibits)."""

from __future__ import annotations

from searchprobe.judging.engine import RunJudgments
from searchprobe.metrics.scorecard import LOWER_IS_BETTER, Scorecard

METRIC_LABELS = {
    "violation_at_k": "Violation@k ↓",
    "satisfied_at_k": "All-constraints-satisfied@k ↑",
    "precision_at_k": "Precision@k ↑",
    "ndcg_at_k": "nDCG@k ↑",
    "reciprocal_rank": "MRR ↑",
}


def failure_exhibits(run: RunJudgments, k: int = 10, limit: int = 10) -> list[dict]:
    """Highest-ranked violating documents, with the evidence that convicts them."""
    exhibits = []
    for jl in run.lists:
        for rank, j in enumerate(jl.judgments[:k], 1):
            if j is None or not j.is_violating(jl.case):
                continue
            violated = [v for v in j.verdicts if v.verdict == "violated"]
            constraint = jl.case.constraint(violated[0].constraint_id) if violated else None
            exhibits.append(
                {
                    "rank": rank,
                    "system": jl.system,
                    "category": jl.case.category.value,
                    "query": jl.case.text,
                    "url": j.url,
                    "constraint": constraint.describe() if constraint else "",
                    "evidence": violated[0].evidence if violated else "",
                    "source": violated[0].source if violated else "",
                }
            )
    exhibits.sort(key=lambda e: (e["rank"], e["system"]))
    return exhibits[:limit]


def scorecard_markdown(
    card: Scorecard,
    run: RunJudgments | None = None,
    k: int = 10,
    judge_note: str | None = None,
    manifest: dict | None = None,
) -> str:
    lines = ["# SearchProbe Scorecard", ""]
    if judge_note:
        lines += [judge_note, ""]
    if manifest:
        packages = ", ".join(f"{k} {v}" for k, v in manifest.get("packages", {}).items())
        lines += [
            "<details><summary>Run manifest</summary>",
            "",
            f"- SearchProbe {manifest.get('searchprobe_version')} at `{manifest.get('git_sha')}`",
            f"- Created {manifest.get('created_at')}, {manifest.get('n_queries', '?')} queries "
            f"(query set `{manifest.get('query_set_id')}`)",
            f"- Python {manifest.get('python')}; {packages}",
            "",
            "</details>",
            "",
        ]
    systems = sorted(card.by_system)
    metrics = list(next(iter(card.by_system.values())).keys()) if card.by_system else []

    lines += [
        "## Overall",
        "",
        "| System | "
        + " | ".join(METRIC_LABELS.get(m, m) for m in metrics)
        + " | Error rate | Judged coverage |",
        "|---" * (len(metrics) + 3) + "|",
    ]
    for s in systems:
        cells = [
            card.by_system[s][m].fmt(pct=m != "ndcg_at_k" and m != "reciprocal_rank")
            for m in metrics
        ]
        lines.append(
            f"| {s} | "
            + " | ".join(cells)
            + f" | {card.error_rate.get(s, 0):.0%} | {card.coverage.get(s, 0):.0%} |"
        )
    lines += ["", f"k = {k}. Brackets are 95% bootstrap CIs over cases.", ""]

    if "violation_at_k" in metrics:
        lines += [
            "## Violation@k by category",
            "",
            "| Category | " + " | ".join(systems) + " |",
            "|---" * (len(systems) + 1) + "|",
        ]
        for cat in sorted(card.by_category):
            row = [card.by_category[cat].get(s, {}).get("violation_at_k") for s in systems]
            lines.append(
                f"| {cat} | " + " | ".join(c.fmt(pct=True) if c else "-" for c in row) + " |"
            )
        lines.append("")

    sig = [c for c in card.comparisons if c.significant]
    lines += ["## Significant differences (paired by case, BH-corrected)", ""]
    if not sig:
        lines.append("None at α = 0.05.")
    for c in sig:
        better = c.system_a if (c.diff.mean < 0) == (c.metric in LOWER_IS_BETTER) else c.system_b
        lines.append(
            f"- **{METRIC_LABELS.get(c.metric, c.metric)}**: {c.system_a} − {c.system_b} = "
            f"{c.diff.mean:+.3f} [{c.diff.lower:+.3f}, {c.diff.upper:+.3f}], "
            f"p = {c.p_value:.3g}, n = {c.n} → **{better}** is better"
        )
    lines.append("")

    if run is not None:
        exhibits = failure_exhibits(run, k=k)
        if exhibits:
            lines += ["## Failure exhibits", ""]
            for e in exhibits:
                lines += [
                    f"- **{e['system']}**, rank {e['rank']} — “{e['query']}” ({e['category']})",
                    f"  - {e['url']}",
                    f"  - Violates: {e['constraint']}",
                ]
                if e["evidence"]:
                    lines.append(f"  - Evidence ({e['source']}): “{e['evidence'][:300]}”")
            lines.append("")
    return "\n".join(lines)
