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
