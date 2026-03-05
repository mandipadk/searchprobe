"""CLI command for perturbation analysis."""

import asyncio
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from searchprobe.config import get_settings
from searchprobe.storage import Database

console = Console()
app = typer.Typer(help="Analyze search robustness through systematic perturbation.")


@app.callback(invoke_without_command=True)
def perturb(
    run_id: str = typer.Option(
        "latest",
        "--run-id",
        "-r",
        help="Run ID to perturb queries from (or 'latest')",
    ),
    operators: str = typer.Option(
        "word_delete,word_swap",
        "--operators",
        "-o",
        help="Comma-separated perturbation operators",
    ),
    max_variants: int = typer.Option(
        5,
        "--max-variants",
        "-v",
        help="Maximum variants per operator per query",
    ),
    max_queries: Optional[int] = typer.Option(
        None,
        "--max-queries",
        "-n",
        help="Maximum number of queries to perturb",
    ),
    provider: str = typer.Option(
        "exa",
        "--provider",
        "-p",
        help="Provider to test perturbation stability",
    ),
    search_mode: Optional[str] = typer.Option(
        None,
        "--mode",
        help="Search mode (e.g., 'neural', 'auto')",
    ),
) -> None:
    """Systematically perturb queries and measure result stability."""
    settings = get_settings()
    db = Database(settings.database_path)

    # Resolve run ID
    actual_run_id = run_id
    if run_id == "latest":
        actual_run_id = db.get_latest_run_id()

    if not actual_run_id:
        console.print("[red]Error: No runs found.[/red]")
        raise typer.Exit(1)

    console.print(f"[blue]Perturbation analysis for run: {actual_run_id[:8]}...[/blue]")

    # Get queries from run
    search_results = db.get_search_results_for_evaluation(
        run_id=actual_run_id,
        skip_evaluated=False,
        max_results=max_queries,
    )

    # Deduplicate by query_id
    seen_queries: set[str] = set()
    queries = []
    for r in search_results:
        if r["query_id"] not in seen_queries:
            seen_queries.add(r["query_id"])
            queries.append({"text": r["query_text"], "category": r["category"], "query_id": r["query_id"]})

    if not queries:
        console.print("[yellow]No queries found for perturbation.[/yellow]")
        raise typer.Exit(0)

    console.print(f"Found {len(queries)} unique queries to perturb")

    # Parse operators
    from searchprobe.perturbation.operators import PerturbationType

    op_list = []
    for op_str in operators.split(","):
        try:
            op_list.append(PerturbationType(op_str.strip()))
        except ValueError:
            console.print(f"[yellow]Unknown operator: {op_str}[/yellow]")

    if not op_list:
        console.print("[red]No valid operators specified.[/red]")
        raise typer.Exit(1)

    # Initialize provider
    from searchprobe.providers.registry import ProviderRegistry

    try:
        search_provider = ProviderRegistry.get_provider(provider, settings)
    except Exception as e:
        console.print(f"[red]Error initializing provider '{provider}': {e}[/red]")
        raise typer.Exit(1)

    # Run perturbation analysis
    from searchprobe.perturbation.engine import PerturbationEngine

    engine = PerturbationEngine(
        provider=search_provider,
        operators=op_list,
        max_variants_per_operator=max_variants,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Perturbing...", total=len(queries))

        def on_progress(query_text: str, index: int, total: int) -> None:
            progress.update(
                task,
                description=f"Perturbing: {query_text[:40]}...",
                completed=index,
            )

        report = asyncio.run(
            engine.analyze_queries(queries, search_mode, on_progress)
        )

    # Build lookup: original_query_text -> query_id
    query_id_map: dict[str, str] = {q["text"]: q["query_id"] for q in queries}

    # Save results
    for analysis in report.analyses:
        db.add_perturbation_result({
            "run_id": actual_run_id,
            "query_id": query_id_map.get(analysis.original_query),
            "provider": analysis.provider,
            "operator": analysis.perturbation_type,
            "original_query": analysis.original_query,
            "perturbed_query": analysis.perturbed_query,
            "jaccard_similarity": analysis.jaccard_similarity,
            "rbo_score": analysis.rbo_score,
        })

    # Display summary
    console.print(f"\n[green]Completed {len(report.analyses)} perturbation analyses[/green]")
    _display_summary(report)


def _display_summary(report: object) -> None:
    """Display perturbation analysis summary."""
    console.print("\n[bold]Perturbation Stability Summary[/bold]")
    console.print(f"  Mean Jaccard similarity: {report.mean_jaccard:.3f}")
    console.print(f"  Mean RBO score: {report.mean_rbo:.3f}")

    if report.stability_by_operator:
        table = Table(title="Stability by Perturbation Operator")
        table.add_column("Operator", style="cyan")
        table.add_column("Mean Jaccard", justify="right")

        for op, score in sorted(report.stability_by_operator.items(), key=lambda x: x[1]):
            color = "red" if score < 0.3 else "yellow" if score < 0.6 else "green"
            table.add_row(
                op.replace("_", " ").title(),
                f"[{color}]{score:.3f}[/{color}]",
            )

        console.print(table)

    if report.stability_by_category:
        table = Table(title="Stability by Category")
        table.add_column("Category", style="cyan")
        table.add_column("Mean Jaccard", justify="right")

        for cat, score in sorted(report.stability_by_category.items(), key=lambda x: x[1]):
            color = "red" if score < 0.3 else "yellow" if score < 0.6 else "green"
            table.add_row(
                cat.replace("_", " ").title(),
                f"[{color}]{score:.3f}[/{color}]",
            )

        console.print(table)

    if report.sensitivity_maps:
        console.print(f"\n[bold]Sensitivity Maps[/bold] ({len(report.sensitivity_maps)} queries)")
        for sm in report.sensitivity_maps[:5]:
            top_words = sm.get_most_sensitive_words(3)
            words_str = ", ".join(f"'{w}' ({s:.2f})" for w, s in top_words)
            console.print(f"  \"{sm.query[:50]}...\" — Load-bearing: {words_str}")
