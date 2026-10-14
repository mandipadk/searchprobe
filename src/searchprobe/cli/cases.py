"""Manage verifiable test-case datasets (TestCase v2)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from searchprobe.cases.io import DATA_ROOT, load_cases, merge_cases, save_cases
from searchprobe.cases.models import TestCase
from searchprobe.cases.twins import attach_twins
from searchprobe.config import get_settings
from searchprobe.storage import Database

console = Console()
app = typer.Typer(help="Create, inspect, validate and import verifiable test cases.")


@app.command("convert-seeds")
def convert_seeds(
    out: Path = typer.Option(DATA_ROOT / "v1" / "seeds.jsonl", "--out", "-o"),
    twins: bool = typer.Option(True, "--twins/--no-twins", help="Generate minimal-pair twins"),
) -> None:
    """Convert the built-in v1 seed queries into v2 test cases."""
    from searchprobe.cases.convert import query_to_case
    from searchprobe.queries.seeds import get_builtin_seeds

    cases = [
        query_to_case(
            q.text,
            q.category.value,
            q.ground_truth.model_dump(mode="json", exclude_none=True) if q.ground_truth else None,
            difficulty=q.difficulty,
            tier="seed",
            adversarial_reason=q.adversarial_reason,
        )
        for q in get_builtin_seeds()
    ]
    if twins:
        cases = attach_twins(cases)
    n = save_cases(cases, out)
    console.print(f"[green]Wrote {n} cases to {out}[/green]")


@app.command("list")
def list_cases(path: Path = typer.Argument(DATA_ROOT, help="JSONL file or directory")) -> None:
    """Summarize a dataset by category, verifiability and twins."""
    cases = load_cases(path)
    table = Table(title=f"{len(cases)} cases in {path}")
    for col in ("Category", "Cases", "Verifiable", "With answer set", "Twins", "Gold"):
        table.add_column(col, justify="right" if col != "Category" else "left")
    by_cat: dict[str, list[TestCase]] = {}
    for c in cases:
        by_cat.setdefault(c.category.value, []).append(c)
    for cat, cs in sorted(by_cat.items()):
        table.add_row(
            cat,
            str(len(cs)),
            str(sum(c.is_verifiable for c in cs)),
            str(sum(bool(c.answer_set) for c in cs)),
            str(sum(bool(c.twin_of) for c in cs)),
            str(sum(c.validation == "gold" for c in cs)),
        )
    console.print(table)
    gens = Counter(c.provenance.generator for c in cases)
    console.print("Generators: " + ", ".join(f"{g}={n}" for g, n in gens.most_common()))


@app.command("validate")
def validate(path: Path = typer.Argument(DATA_ROOT)) -> None:
    """Check schema validity and dataset integrity (twins resolve, ids unique)."""
    cases = load_cases(path)  # raises with file:line on schema errors
    ids = Counter(c.id for c in cases)
    problems = [f"duplicate id {i} ({n}x)" for i, n in ids.items() if n > 1]
    known = set(ids)
    for c in cases:
        if not c.hard_constraints:
            problems.append(f"{c.id}: no hard constraints")
        for rel, tid in c.twins.items():
            if tid not in known:
                problems.append(f"{c.id}: twin '{rel}' -> {tid} not in dataset")
        if c.twin_of and c.twin_of not in known:
            problems.append(f"{c.id}: twin_of {c.twin_of} not in dataset")
    if problems:
        for p in problems:
            console.print(f"[red]✗[/red] {p}")
        raise typer.Exit(1)
    console.print(f"[green]✓ {len(cases)} cases valid[/green]")


@app.command("twins")
def twins_cmd(
    path: Path = typer.Argument(..., help="JSONL file to augment in place"),
) -> None:
    """Generate missing minimal-pair twins for every case in a file."""
    cases = load_cases(path)
    before = len(cases)
    cases = merge_cases(cases, attach_twins(cases))
    save_cases(cases, path)
    console.print(f"[green]{path}: {before} -> {len(cases)} cases[/green]")


@app.command("import")
def import_cases(
    path: Path = typer.Argument(DATA_ROOT, help="JSONL file or directory"),
    name: str | None = typer.Option(None, "--name", help="Query set name"),
    category: str | None = typer.Option(None, "--categories", "-c", help="Comma-separated filter"),
    limit: int | None = typer.Option(None, "--limit", "-n"),
) -> None:
    """Import cases into the database as a query set (then `searchprobe run`)."""
    cases = load_cases(path)
    if category:
        wanted = {c.strip() for c in category.split(",")}
        cases = [c for c in cases if c.category.value in wanted]
    if limit:
        cases = cases[:limit]
    if not cases:
        console.print("[yellow]No cases selected.[/yellow]")
        raise typer.Exit(1)
    db = Database(get_settings().database_path)
    qs_id = db.add_cases(cases, name=name or f"cases:{Path(path).name}")
    console.print(f"[green]Imported {len(cases)} cases as query set {qs_id}[/green]")
    console.print("Next: searchprobe run --query-set latest")


generate_app = typer.Typer(help="Generate verifiable cases.")
app.add_typer(generate_app, name="generate")


@generate_app.command("wikidata")
def generate_wikidata(
    out: Path = typer.Option(DATA_ROOT / "v1" / "wikidata.jsonl", "--out", "-o"),
    only: str | None = typer.Option(None, "--only", help="Comma-separated spec name prefixes"),
) -> None:
    """Build cases whose answers come from Wikidata (answer + negative sets, twins)."""
    import asyncio

    from searchprobe.cases.generators.wikidata import default_specs, generate

    specs = default_specs()
    if only:
        prefixes = tuple(p.strip() for p in only.split(","))
        specs = [s for s in specs if s.name.startswith(prefixes)]
        if not specs:
            console.print(f"[red]No generator matches --only {only!r}[/red]")
            raise typer.Exit(1)
    console.print(f"Running {len(specs)} Wikidata generators...")
    cases, failures = asyncio.run(
        generate(specs, on_progress=lambda name, n: console.print(f"  {name}: {n} case(s)"))
    )
    existing = load_cases(out) if out.exists() else []
    n = save_cases(merge_cases(existing, cases), out)
    console.print(f"[green]{len(cases)} cases generated; {out} now holds {n}[/green]")
    for f in failures:
        console.print(f"[yellow]  {f}[/yellow]")


@generate_app.command("structured")
def generate_structured_cmd(
    out: Path = typer.Option(DATA_ROOT / "v1" / "structured.jsonl", "--out", "-o"),
) -> None:
    """Build template cases with mechanically checkable constraints (language, pub date)."""
    from searchprobe.cases.generators.structured import generate_structured

    cases = generate_structured()
    n = save_cases(cases, out)
    console.print(f"[green]Wrote {n} cases to {out}[/green]")


@generate_app.command("llm")
def generate_llm(
    categories: str = typer.Option(..., "--categories", "-c", help="Comma-separated categories"),
    count: int = typer.Option(10, "--count", "-n", help="Cases per category"),
    out: Path = typer.Option(DATA_ROOT / "v1" / "llm.jsonl", "--out", "-o"),
    answerable_with: str | None = typer.Option(
        None, "--answerable-with", help="Keep only cases this provider can answer (e.g. exa)"
    ),
) -> None:
    """Generate LLM cases with typed constraints and twins (validated; optionally answerable)."""
    import asyncio

    from searchprobe.cases.generators.llm import check_answerable, generate_llm_cases
    from searchprobe.llm import get_async_client
    from searchprobe.queries.taxonomy import AdversarialCategory

    settings = get_settings()
    if not settings.has_anthropic_configured():
        console.print("[red]Anthropic credentials required.[/red]")
        raise typer.Exit(1)
    client = get_async_client(settings)
    existing = load_cases(out) if out.exists() else []

    async def run() -> list[TestCase]:
        produced: list[TestCase] = []
        for cat in [AdversarialCategory(c.strip()) for c in categories.split(",")]:
            avoid = [c.text for c in existing if c.category == cat]
            cases, rejected = await generate_llm_cases(
                client, settings.generation_model, cat, count, avoid, settings.generation_effort
            )
            console.print(f"  {cat.value}: {len(cases)} cases, {len(rejected)} rejected")
            for r in rejected:
                console.print(f"[dim]    rejected {r}[/dim]")
            produced += cases
        if not answerable_with:
            return produced
        from searchprobe.evidence import EvidenceFetcher, EvidenceStore
        from searchprobe.judging.llm_judge import DocumentJudge
        from searchprobe.providers.registry import ProviderRegistry

        provider = ProviderRegistry.get_provider(answerable_with, settings)
        judge = DocumentJudge(client, settings.judge_model, settings.judge_effort)
        kept: list[TestCase] = []
        async with EvidenceFetcher(EvidenceStore(settings.database_path)) as fetcher:
            for case in produced:
                if case.twin_of:
                    kept.append(case)  # twins are kept with their base below
                    continue
                ok, url = await check_answerable(case, provider, judge, fetcher)
                if ok:
                    kept.append(case.model_copy(update={"known_positive_urls": [url]}))
                else:
                    console.print(f"[dim]    unanswerable, dropped: {case.text}[/dim]")
        bases = {c.id for c in kept if not c.twin_of}
        return [c for c in kept if not c.twin_of or c.twin_of in bases]

    cases = asyncio.run(run())
    n = save_cases(merge_cases(existing, cases), out)
    console.print(f"[green]{len(cases)} cases added; {out} now holds {n}[/green]")


@app.command("review")
def review(
    path: Path = typer.Argument(..., help="JSONL file to review in place"),
    category: str | None = typer.Option(None, "--category", "-c"),
) -> None:
    """Interactively promote cases to gold (g), drop (d), skip (s) or quit (q)."""
    cases = load_cases(path)
    by_id = {c.id: c for c in cases}
    queue = [
        c
        for c in cases
        if c.validation != "gold"
        and not c.twin_of
        and (category is None or c.category.value == category)
    ]
    for i, case in enumerate(queue, 1):
        console.rule(f"{i}/{len(queue)} {case.category.value} ({case.provenance.generator})")
        console.print(f"[bold]{case.text}[/bold]")
        for c in case.constraints:
            console.print(f"  {c.id} [{c.type}{'' if c.hard else ', soft'}] {c.describe()}")
        if case.answer_set:
            sample = ", ".join(e.label for e in case.answer_set[:12])
            console.print(f"  answers ({len(case.answer_set)}): {sample}")
        if case.negative_set:
            console.print(f"  negatives: {', '.join(e.label for e in case.negative_set[:8])}")
        for rel, tid in case.twins.items():
            if tid in by_id:
                console.print(f"  twin [{rel}]: {by_id[tid].text}")
        ans = typer.prompt("  (g)old / (d)rop / (s)kip / (q)uit", default="s").strip().lower()
        if ans == "q":
            break
        if ans == "g":
            by_id[case.id] = case.model_copy(update={"validation": "gold"})
            for tid in case.twins.values():
                if tid in by_id:
                    by_id[tid] = by_id[tid].model_copy(update={"validation": "gold"})
        elif ans == "d":
            by_id.pop(case.id, None)
            for tid in case.twins.values():
                by_id.pop(tid, None)
    save_cases(list(by_id.values()), path)
    console.print(f"[green]Saved {len(by_id)} cases to {path}[/green]")
