"""Metamorphic testing: does the engine change its results when the meaning changes?"""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from searchprobe.cases.io import DATA_ROOT, load_cases
from searchprobe.config import get_settings

console = Console()
app = typer.Typer(help="Operator Blindness Index from case/twin pairs (no ground truth needed).")


def build_providers(systems: list[str], settings) -> dict:
    from searchprobe.pipeline.rate_limiter import RateLimiterPool
    from searchprobe.providers.registry import ProviderRegistry
    from searchprobe.providers.resilient import ResilientProvider

    limiter = RateLimiterPool()
    providers = {}
    for key in systems:
        name = key.split(":", 1)[0]
        raw = ProviderRegistry.get_provider(name, settings)
        providers[key] = ResilientProvider(raw, before_attempt=lambda n=name: limiter.acquire(n))
    return providers


@app.callback(invoke_without_command=True)
def metamorphic(
    systems: str = typer.Option(..., "--systems", "-s", help="e.g. exa:neural,exa:auto,brave"),
    cases_path: Path = typer.Option(DATA_ROOT, "--cases", help="JSONL file or directory"),
    categories: str | None = typer.Option(None, "--categories", "-c"),
    limit: int | None = typer.Option(None, "--limit", "-n", help="Max base cases"),
    repeats: int = typer.Option(1, "--repeats", help="Re-issues per query (noise floor)"),
    typos: bool = typer.Option(True, "--typos/--no-typos", help="Add typo INV variants"),
    paraphrases: int = typer.Option(0, "--paraphrases", help="LLM paraphrases per case (INV)"),
    judge_direction: bool = typer.Option(
        False, "--judge-direction", help="Judge whether DIR results moved correctly"
    ),
    budget: float = typer.Option(5.0, "--budget", "-b"),
    out: Path | None = typer.Option(None, "--out", "-o", help="Write Markdown report"),
) -> None:
    """Run base/twin/repeat/paraphrase searches and report OBI with bootstrap CIs."""
    from searchprobe.metamorphic.engine import MetamorphicEngine
    from searchprobe.metamorphic.obi import compute_obi, summarize
    from searchprobe.metamorphic.report import render_markdown, save_run
    from searchprobe.metamorphic.variants import Variant, build_variants, llm_paraphrases
    from searchprobe.pipeline.cost_tracker import CostTracker

    settings = get_settings()
    cases = load_cases(cases_path)
    if categories:
        wanted = {c.strip() for c in categories.split(",")}
        keep = {c.id for c in cases if c.category.value in wanted}
        cases = [c for c in cases if c.id in keep or (c.twin_of in keep)]
    bases = [c for c in cases if not c.twin_of and c.twins]
    if limit:
        bases = bases[:limit]
        ids = {c.id for c in bases} | {t for c in bases for t in c.twins.values()}
        cases = [c for c in cases if c.id in ids]
    variants = build_variants(cases, repeats=repeats, typos=typos)
    if not variants:
        console.print("[yellow]No cases with twins selected.[/yellow]")
        raise typer.Exit(1)

    client = None
    if paraphrases or judge_direction:
        from searchprobe.llm import get_async_client

        client = get_async_client(settings)
    if paraphrases:

        async def para() -> list:
            out_v = []
            for c in bases:
                for p in await llm_paraphrases(client, settings.generation_model, c, paraphrases):
                    out_v.append(Variant(c.id, "INV", "paraphrase", p))
            return out_v

        variants += asyncio.run(para())

    system_list = [s.strip() for s in systems.split(",")]
    engine = MetamorphicEngine(
        build_providers(system_list, settings), cost_tracker=CostTracker(budget_limit=budget)
    )
    console.print(f"{len(bases)} base cases, {len(variants)} variants, {len(system_list)} systems")

    async def go():
        run = await engine.run(cases, variants)
        if judge_direction:
            from searchprobe.evidence import EvidenceFetcher, EvidenceStore
            from searchprobe.judging.llm_judge import DocumentJudge

            judge = DocumentJudge(client, settings.judge_model, settings.judge_effort)
            async with EvidenceFetcher(EvidenceStore(settings.database_path)) as fetcher:
                await engine.judge_directions(run, cases, variants, judge, fetcher)
        return run

    run = asyncio.run(go())
    run_id = save_run(settings.database_path, run)
    dir_rows, inv_rows = compute_obi(run.results)

    table = Table(title="Operator Blindness Index (lower is better; 1.0 = operator ignored)")
    table.add_column("System")
    table.add_column("Relation")
    table.add_column("OBI [95% CI]", justify="right")
    table.add_column("n", justify="right")
    for s in summarize(dir_rows):
        table.add_row(
            s.system, s.group, f"{s.obi.mean:.2f} [{s.obi.lower:.2f}, {s.obi.upper:.2f}]", str(s.n)
        )
    console.print(table)
    console.print(
        f"Saved metamorphic run {run_id}. Searches: {run.searches}, cost ${run.cost_usd:.4f}"
    )
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_markdown(run))
        console.print(f"[green]Wrote {out}[/green]")
