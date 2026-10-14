"""Explain failures: triplet probes on open retrievers, linked per case to providers."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from searchprobe.cases.io import DATA_ROOT, load_cases
from searchprobe.config import get_settings

console = Console()
app = typer.Typer(help="Do retrievers prefer satisfying over violating documents? (triplets)")


@app.command("triplets")
def build_triplets(
    cases_path: Path = typer.Option(DATA_ROOT, "--cases"),
    out: Path = typer.Option(Path("reports/triplets.jsonl"), "--out", "-o"),
    per_case: int = typer.Option(5, "--per-case"),
    from_judgments: bool = typer.Option(
        True, "--from-judgments/--no-judgments", help="Also use judged documents from runs"
    ),
) -> None:
    """Build triplets from known answer/negative sets and from judged result pools."""
    from searchprobe.explain.triplets import (
        WikipediaSummaries,
        save_triplets,
        triplets_from_judgments,
        triplets_from_known_sets,
    )

    cases = load_cases(cases_path)

    async def known() -> list:
        summaries = WikipediaSummaries()
        try:
            return await triplets_from_known_sets(cases, summaries, per_case=per_case)
        finally:
            await summaries.close()

    triplets = asyncio.run(known())
    console.print(f"{len(triplets)} triplets from known sets")
    if from_judgments:
        from searchprobe.cases.convert import db_query_to_case
        from searchprobe.evidence.store import EvidenceStore
        from searchprobe.judging.store import JudgmentStore
        from searchprobe.storage import Database

        settings = get_settings()
        db = Database(settings.database_path)
        db_cases = {}
        for row in db.get_queries():
            c = db_query_to_case(row)
            db_cases[c.id] = c
        judgments = JudgmentStore(settings.database_path).all()
        store = EvidenceStore(settings.database_path)
        texts = {}
        for j in judgments:
            snap = store.get(j.content_hash)
            if snap is not None and snap.ok:
                texts[j.content_hash] = snap.text
        judged = triplets_from_judgments(db_cases, judgments, texts, per_case=per_case)
        console.print(f"{len(judged)} triplets from judged result pools")
        triplets += judged
    save_triplets(triplets, out)
    console.print(f"[green]Wrote {len(triplets)} triplets to {out}[/green]")


@app.command("probe")
def probe(
    triplets_path: Path = typer.Option(Path("reports/triplets.jsonl"), "--triplets"),
    scorers: str = typer.Option(
        "bm25",
        "--scorers",
        help="Comma-separated: bm25, st:<model>, ce:<model>, openai:<model>. Example: "
        "bm25,st:BAAI/bge-small-en-v1.5,st:intfloat/e5-base-v2,"
        "ce:cross-encoder/ms-marco-MiniLM-L-6-v2",
    ),
    run_id: str | None = typer.Option(
        None, "--run-id", help="Link to a judged run: AUC of triplet failure -> provider violation"
    ),
    out: Path | None = typer.Option(None, "--out", "-o"),
) -> None:
    """Score triplets with each scorer; optionally predict provider failures per case."""
    from searchprobe.explain.probe import (
        predict_failures,
        probe_markdown,
        run_probe,
        summarize_probe,
    )
    from searchprobe.explain.scorers import make_scorer
    from searchprobe.explain.triplets import load_triplets

    settings = get_settings()
    triplets = load_triplets(triplets_path)
    built = []
    for spec in [s.strip() for s in scorers.split(",") if s.strip()]:
        try:
            built.append(make_scorer(spec, settings))
        except ImportError as e:
            console.print(
                f"[yellow]Skipping {spec}: {e} (pip install 'searchprobe[analysis]')[/yellow]"
            )
    scores = run_probe(triplets, built)

    predictions = None
    if run_id:
        from searchprobe.cli.judge import _judge, _resolve_run, build_engine
        from searchprobe.metrics.scorecard import compute_list_metrics
        from searchprobe.storage import Database

        actual = _resolve_run(Database(settings.database_path), run_id)
        engine, fetcher = build_engine(
            settings, allow_llm=False, fetch=False, budget=None, concurrency=8
        )
        run = asyncio.run(_judge(engine, fetcher, actual, None, 10, False, None))
        violated = {
            (m.system, m.case_id): m.any_violation
            for m in compute_list_metrics(run)
            if m.error is None and m.n_judged
        }
        predictions = predict_failures(scores, violated)

    table = Table(title="Pairwise accuracy (satisfying > violating; 0.5 = chance)")
    table.add_column("Scorer")
    table.add_column("Group")
    table.add_column("Accuracy [95% CI]", justify="right")
    table.add_column("n", justify="right")
    for s in summarize_probe(scores):
        if s.group == "ALL" or len(triplets) < 400:
            table.add_row(
                s.scorer,
                s.group,
                f"{s.accuracy.mean:.0%} [{s.accuracy.lower:.0%}, {s.accuracy.upper:.0%}]",
                str(s.n),
            )
    console.print(table)
    if predictions:
        for p in predictions:
            console.print(
                f"  {p.scorer} → {p.system}: AUC {p.auc:.2f} "
                f"[{p.lower:.2f}, {p.upper:.2f}] (n={p.n_cases})"
            )
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(probe_markdown(scores, predictions))
        console.print(f"[green]Wrote {out}[/green]")
