"""Discover new failure cases with MAP-Elites and differential fitness."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from searchprobe.cases.io import DATA_ROOT, load_cases, merge_cases, save_cases
from searchprobe.config import get_settings

console = Console()
app = typer.Typer(help="Breed harder, still-answerable variants of verifiable cases.")


@app.callback(invoke_without_command=True)
def discover(
    target: str = typer.Option(..., "--target", "-t", help="System to break, e.g. exa:neural"),
    reference: str | None = typer.Option(
        None, "--reference", "-r", help="System that must answer the case, e.g. brave"
    ),
    cases_path: Path = typer.Option(DATA_ROOT / "v1" / "wikidata.jsonl", "--cases"),
    categories: str | None = typer.Option(None, "--categories", "-c"),
    seeds: int = typer.Option(10, "--seeds", help="Number of seed cases"),
    iterations: int = typer.Option(40, "--iterations", "-n"),
    k: int = typer.Option(5, "--k"),
    llm_mutations: bool = typer.Option(True, "--llm/--no-llm", help="Use LLM rephrasings"),
    budget: float = typer.Option(10.0, "--budget", "-b"),
    min_fitness: float = typer.Option(0.2, "--min-fitness"),
    out: Path = typer.Option(DATA_ROOT / "v1" / "discovered.jsonl", "--out", "-o"),
) -> None:
    """Search for case variants the target fails on while the reference succeeds."""
    from searchprobe.cli.metamorphic import build_providers
    from searchprobe.discovery.search import CaseEvaluator, promote
    from searchprobe.discovery.search import discover as run_discovery
    from searchprobe.evidence import EvidenceFetcher, EvidenceStore
    from searchprobe.judging.llm_judge import DocumentJudge
    from searchprobe.llm import get_async_client
    from searchprobe.pipeline.cost_tracker import CostTracker

    settings = get_settings()
    if not settings.has_anthropic_configured():
        console.print("[red]Anthropic credentials required (the judge decides fitness).[/red]")
        raise typer.Exit(1)
    pool = [c for c in load_cases(cases_path) if not c.twin_of]
    if categories:
        wanted = {c.strip() for c in categories.split(",")}
        pool = [c for c in pool if c.category.value in wanted]
    seed_cases = pool[:seeds]
    systems = [target] + ([reference] if reference else [])
    client = get_async_client(settings)

    async def go():
        async with EvidenceFetcher(EvidenceStore(settings.database_path)) as fetcher:
            evaluator = CaseEvaluator(
                build_providers(systems, settings),
                target,
                reference,
                DocumentJudge(client, settings.judge_model, settings.judge_effort),
                fetcher=fetcher,
                k=k,
                cost_tracker=CostTracker(budget_limit=budget),
            )
            return await run_discovery(
                seed_cases,
                evaluator,
                iterations=iterations,
                llm=(client, settings.generation_model) if llm_mutations else None,
                progress=lambda i, n, best: console.print(f"  iter {i}/{n} best={best:.2f}"),
            ), evaluator

    result, evaluator = asyncio.run(go())
    table = Table(title=f"Archive: {len(result.archive.cells)} cells, {result.evaluated} evaluated")
    for col in ("Fitness", "Re-eval mean", "Cell", "Query"):
        table.add_column(col)
    for e in result.archive.elites()[:20]:
        table.add_row(
            f"{e.evaluation.fitness:.2f}",
            f"{e.mean_fitness:.2f}",
            "/".join(e.cell),
            e.case.text[:70],
        )
    console.print(table)
    if evaluator.budget_exhausted:
        console.print("[yellow]Budget exhausted.[/yellow]")
    promoted = promote(result, min_fitness)
    existing = load_cases(out) if out.exists() else []
    n = save_cases(merge_cases(existing, promoted), out)
    console.print(f"[green]Promoted {len(promoted)} cases (incl. twins); {out} holds {n}[/green]")
