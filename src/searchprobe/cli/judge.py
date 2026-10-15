"""Per-document judging and scorecards for benchmark runs."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from searchprobe.config import get_settings
from searchprobe.storage import Database

console = Console()
judge_app = typer.Typer(help="Judge every returned document against each case's constraints.")
scorecard_app = typer.Typer(help="Compute a scorecard (IR metrics + CIs) from stored judgments.")


def _resolve_run(db: Database, run_id: str) -> str:
    actual = db.get_latest_run_id() if run_id == "latest" else run_id
    if not actual:
        console.print("[red]No runs found. Run `searchprobe run` first.[/red]")
        raise typer.Exit(1)
    return actual


def build_engine(settings, *, allow_llm: bool, fetch: bool, budget: float | None, concurrency: int):
    """Construct the judging engine from settings."""
    from searchprobe.evidence import EvidenceFetcher, EvidenceStore
    from searchprobe.judging.engine import JudgingEngine
    from searchprobe.judging.llm_judge import DocumentJudge
    from searchprobe.judging.store import JudgmentStore
    from searchprobe.pipeline.cost_tracker import CostTracker

    db = Database(settings.database_path)
    client = None
    if allow_llm:
        from searchprobe.llm import get_async_client

        client = get_async_client(settings)
    judge = DocumentJudge(client, settings.judge_model, settings.judge_effort)
    store = EvidenceStore(settings.database_path)
    if fetch:
        fetcher = EvidenceFetcher(store)
    else:
        # Offline: reuse stored snapshots so content hashes (and cached judgments) match.
        fetcher = EvidenceFetcher(store, offline=True, max_age_days=None)
    engine = JudgingEngine(
        db,
        JudgmentStore(settings.database_path),
        judge,
        fetcher=fetcher,
        cost_tracker=CostTracker(budget_limit=budget) if budget is not None else None,
        concurrency=concurrency,
        allow_llm=allow_llm,
    )
    return engine, fetcher


async def _judge(engine, fetcher, run_id, systems, k, refresh, progress_cb):
    try:
        return await engine.judge_run(
            run_id, systems=systems, k=k, refresh_evidence=refresh, progress=progress_cb
        )
    finally:
        if fetcher is not None:
            await fetcher.close()


def _print_scorecard(card, k: int) -> None:
    from searchprobe.metrics.render import METRIC_LABELS

    metrics = list(next(iter(card.by_system.values())).keys()) if card.by_system else []
    table = Table(title=f"Scorecard (k={k}, 95% bootstrap CIs)")
    table.add_column("System", style="cyan")
    for m in metrics:
        table.add_column(METRIC_LABELS.get(m, m), justify="right")
    table.add_column("Errors", justify="right")
    for s, cells in sorted(card.by_system.items()):
        table.add_row(
            s,
            *[
                cells[m].fmt(pct=m in ("violation_at_k", "satisfied_at_k", "precision_at_k"))
                for m in metrics
            ],
            f"{card.error_rate.get(s, 0):.0%}",
        )
    console.print(table)
    sig = [c for c in card.comparisons if c.significant]
    if sig:
        console.print("[bold]Significant paired differences (BH-corrected):[/bold]")
        for c in sig:
            console.print(
                f"  {METRIC_LABELS.get(c.metric, c.metric)}: {c.system_a} − {c.system_b} = "
                f"{c.diff.mean:+.3f} [{c.diff.lower:+.3f}, {c.diff.upper:+.3f}] (n={c.n})"
            )


def _emit(card, run, k, out: Path | None, fmt: str, judge_note: str, manifest=None) -> None:
    from searchprobe.metrics.render import scorecard_markdown

    if fmt in ("table", "both"):
        _print_scorecard(card, k)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix == ".json":
            out.write_text(json.dumps(card.to_dict(), indent=2, default=str))
        else:
            out.write_text(
                scorecard_markdown(card, run, k=k, judge_note=judge_note, manifest=manifest)
            )
        console.print(f"[green]Wrote {out}[/green]")
    elif fmt in ("markdown", "both"):
        console.print(scorecard_markdown(card, run, k=k, judge_note=judge_note, manifest=manifest))


@judge_app.callback(invoke_without_command=True)
def judge(
    run_id: str = typer.Option("latest", "--run-id", "-r"),
    systems: str | None = typer.Option(None, "--systems", "-s", help="e.g. exa:neural,brave"),
    k: int = typer.Option(10, "--k", help="Judge the top-k results of each list"),
    budget: float = typer.Option(5.0, "--budget", "-b", help="Max judge spend in USD"),
    fetch: bool = typer.Option(True, "--fetch/--no-fetch", help="Fetch pages (else snippets)"),
    refresh: bool = typer.Option(False, "--refresh", help="Re-fetch pages even if cached"),
    concurrency: int = typer.Option(8, "--concurrency"),
    out: Path | None = typer.Option(None, "--out", "-o", help="Write scorecard (.md or .json)"),
) -> None:
    """Fetch evidence and judge each (case, document) pair, then print a scorecard."""
    from searchprobe.metrics.scorecard import build_scorecard, compute_list_metrics

    settings = get_settings()
    if not settings.has_anthropic_configured():
        console.print("[red]Anthropic credentials required for judging.[/red]")
        raise typer.Exit(1)
    db = Database(settings.database_path)
    actual = _resolve_run(db, run_id)
    engine, fetcher = build_engine(
        settings, allow_llm=True, fetch=fetch, budget=budget, concurrency=concurrency
    )
    system_list = [s.strip() for s in systems.split(",")] if systems else None

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Judging documents...", total=None)

        def cb(done: int, total: int) -> None:
            progress.update(task, completed=done, total=total)

        run = asyncio.run(_judge(engine, fetcher, actual, system_list, k, refresh, cb))

    console.print(
        f"Judged {run.judged_pairs} new pairs, {run.cached_pairs} cached, "
        f"{run.checker_only_pairs} decided by checkers, {run.failed_pairs} failed, "
        f"{run.skipped_for_budget} skipped for budget. Cost ${run.cost_usd:.4f}"
    )
    card = build_scorecard(compute_list_metrics(run, k=k))
    manifest = ((db.get_run(actual) or {}).get("config") or {}).get("manifest")
    _emit(card, run, k, out, "table", f"Judge: {engine.judge.judge_id}", manifest)


@scorecard_app.callback(invoke_without_command=True)
def scorecard(
    run_id: str = typer.Option("latest", "--run-id", "-r"),
    systems: str | None = typer.Option(None, "--systems", "-s"),
    k: int = typer.Option(10, "--k"),
    fmt: str = typer.Option("table", "--format", "-f", help="table | markdown | both"),
    out: Path | None = typer.Option(None, "--out", "-o", help="Write scorecard (.md or .json)"),
) -> None:
    """Recompute the scorecard from cached judgments (no LLM calls, no fetching)."""
    from searchprobe.metrics.scorecard import build_scorecard, compute_list_metrics

    settings = get_settings()
    db = Database(settings.database_path)
    actual = _resolve_run(db, run_id)
    engine, fetcher = build_engine(
        settings, allow_llm=False, fetch=False, budget=None, concurrency=8
    )
    system_list = [s.strip() for s in systems.split(",")] if systems else None
    run = asyncio.run(_judge(engine, fetcher, actual, system_list, k, False, None))
    card = build_scorecard(compute_list_metrics(run, k=k))
    manifest = ((db.get_run(actual) or {}).get("config") or {}).get("manifest")
    _emit(
        card, run, k, out, fmt, f"Judge: {engine.judge.judge_id} (cached judgments only)", manifest
    )
