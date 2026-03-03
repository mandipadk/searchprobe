"""Evaluate benchmark command."""

import asyncio
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from searchprobe.config import get_settings
from searchprobe.evaluation.judge import SearchJudge
from searchprobe.evaluation.statistics import summary_statistics, compare_providers
from searchprobe.providers.models import SearchResponse, SearchResult
from searchprobe.storage import Database

console = Console()
app = typer.Typer(help="Evaluate benchmark results using LLM-as-judge.")


@app.callback(invoke_without_command=True)
def evaluate(
    run_id: str = typer.Option(
        "latest",
        "--run-id",
        "-r",
        help="Run ID to evaluate (or 'latest')",
    ),
    max_results: Optional[int] = typer.Option(
        None,
        "--max-results",
        "-n",
        help="Maximum number of results to evaluate",
    ),
    categories: Optional[str] = typer.Option(
        None,
        "--categories",
        "-c",
        help="Comma-separated categories to evaluate (default: all)",
    ),
    providers: Optional[str] = typer.Option(
        None,
        "--providers",
        "-p",
        help="Comma-separated providers to evaluate (default: all)",
    ),
    skip_evaluated: bool = typer.Option(
        True,
        "--skip-evaluated/--re-evaluate",
        help="Skip results that have already been evaluated",
    ),
    show_comparisons: bool = typer.Option(
        True,
        "--comparisons/--no-comparisons",
        help="Show provider comparisons",
    ),
) -> None:
    """Evaluate search results using LLM-as-judge."""
    settings = get_settings()

    # Check for API key
    if not settings.anthropic_api_key:
        console.print("[red]Error: Anthropic API key required for evaluation.[/red]")
        console.print("Set SEARCHPROBE_ANTHROPIC_API_KEY in .env file")
        raise typer.Exit(1)

    db = Database(settings.database_path)

    # Resolve run ID
    actual_run_id = run_id
    if run_id == "latest":
        actual_run_id = db.get_latest_run_id()

    if not actual_run_id:
        console.print("[red]Error: No runs found to evaluate.[/red]")
        console.print("Run 'searchprobe run' first to create benchmark results.")
        raise typer.Exit(1)

    console.print(f"[blue]Evaluating run: {actual_run_id[:8]}...[/blue]")

    # Get search results to evaluate
    results_to_eval = db.get_search_results_for_evaluation(
        run_id=actual_run_id,
        skip_evaluated=skip_evaluated,
        max_results=max_results,
    )

    # Filter by category if specified
    if categories:
        category_list = [c.strip().lower() for c in categories.split(",")]
        results_to_eval = [
            r for r in results_to_eval if r["category"] in category_list
        ]

    # Filter by provider if specified
    if providers:
        provider_list = [p.strip().lower() for p in providers.split(",")]
        results_to_eval = [
            r for r in results_to_eval if r["provider"] in provider_list
        ]

    if not results_to_eval:
        console.print("[yellow]No results to evaluate.[/yellow]")
        if skip_evaluated:
            console.print("Use --re-evaluate to re-evaluate already evaluated results.")
        raise typer.Exit(0)

    console.print(f"Found {len(results_to_eval)} results to evaluate")

    # Initialize judge
    judge = SearchJudge(settings)

    # Evaluate with progress
    evaluated = 0
    all_evaluations = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Evaluating...", total=len(results_to_eval))

        for result_item in results_to_eval:
            progress.update(
                task,
                description=f"Evaluating: {result_item['query_text'][:40]}...",
            )

            # Reconstruct SearchResponse from stored data
            search_response = _reconstruct_response(result_item)

            # Evaluate
            evaluation = asyncio.run(
                judge.evaluate(
                    query_id=result_item["query_id"],
                    query_text=result_item["query_text"],
                    category=result_item["category"],
                    search_response=search_response,
                    ground_truth=result_item.get("ground_truth"),
                )
            )

            # Store evaluation
            db.add_evaluation(actual_run_id, evaluation.to_dict())
            all_evaluations.append(evaluation.to_dict())

            evaluated += 1
            progress.update(task, completed=evaluated)

    console.print(f"\n[green]Evaluated {evaluated} results[/green]")

    # Show summary
    _display_summary(all_evaluations, show_comparisons)


def _reconstruct_response(result_item: dict) -> SearchResponse:
    """Reconstruct SearchResponse from database record."""
    results_data = result_item.get("results", [])

    search_results = []
    for r in results_data:
        search_results.append(
            SearchResult(
                title=r.get("title", ""),
                url=r.get("url", "http://example.com"),
                snippet=r.get("snippet", ""),
                content=r.get("content"),
                score=r.get("score"),
                published_date=r.get("published_date"),
            )
        )

    return SearchResponse(
        provider=result_item.get("provider", "unknown"),
        search_mode=result_item.get("search_mode"),
        query=result_item.get("query_text", ""),
        results=search_results,
        latency_ms=result_item.get("latency_ms", 0),
        cost_usd=result_item.get("cost_usd", 0),
        error=result_item.get("error"),
    )


def _display_summary(evaluations: list[dict], show_comparisons: bool) -> None:
    """Display evaluation summary."""
    if not evaluations:
        return

    stats = summary_statistics(evaluations)

    # Overall stats
    console.print("\n[bold]Evaluation Summary[/bold]")
    console.print(f"  Total evaluated: {stats['count']}")
    console.print(f"  Mean score: {stats['mean']:.3f}")
    console.print(f"  Median score: {stats['median']:.3f}")
    console.print(f"  Std deviation: {stats['std']:.3f}")
    console.print(f"  95% CI: {stats['ci_95']}")

    # By provider table
    if stats.get("by_provider"):
        table = Table(title="Scores by Provider")
        table.add_column("Provider", style="cyan")
        table.add_column("Mode", style="blue")
        table.add_column("Mean", justify="right")
        table.add_column("95% CI", justify="right")
        table.add_column("n", justify="right")

        for provider_key, ci in stats["by_provider"].items():
            parts = provider_key.split(":", 1)
            provider = parts[0]
            mode = parts[1] if len(parts) > 1 else "-"

            table.add_row(
                provider,
                mode,
                f"{ci.mean:.3f}",
                f"[{ci.lower:.3f}, {ci.upper:.3f}]",
                str(ci.n),
            )

        console.print(table)

    # By category table
    if stats.get("by_category"):
        table = Table(title="Scores by Category")
        table.add_column("Category", style="cyan")

        # Get all providers
        all_providers = set()
        for cat_data in stats["by_category"].values():
            all_providers.update(cat_data.keys())

        for provider in sorted(all_providers):
            table.add_column(provider, justify="right")

        for category, provider_scores in stats["by_category"].items():
            row = [category]
            for provider in sorted(all_providers):
                if provider in provider_scores:
                    ci = provider_scores[provider]
                    row.append(f"{ci.mean:.2f}")
                else:
                    row.append("-")
            table.add_row(*row)

        console.print(table)

    # Failure modes
    if stats.get("failure_modes"):
        console.print("\n[bold]Top Failure Modes[/bold]")
        for mode, count in list(stats["failure_modes"].items())[:10]:
            console.print(f"  {mode}: {count}")

    # Provider comparisons
    if show_comparisons and len(stats.get("by_provider", {})) >= 2:
        console.print("\n[bold]Provider Comparisons[/bold]")

        providers = list(stats["by_provider"].keys())
        for i, prov_a in enumerate(providers):
            for prov_b in providers[i + 1:]:
                # Get scores for comparison
                scores_a = [
                    e["weighted_score"]
                    for e in evaluations
                    if f"{e['provider']}:{e.get('search_mode', 'default')}" == prov_a
                ]
                scores_b = [
                    e["weighted_score"]
                    for e in evaluations
                    if f"{e['provider']}:{e.get('search_mode', 'default')}" == prov_b
                ]

                if scores_a and scores_b:
                    comparison = compare_providers(
                        prov_a, scores_a, prov_b, scores_b, paired=False
                    )

                    sig_marker = "*" if comparison.significant else ""
                    winner = prov_a if comparison.mean_diff > 0 else prov_b
                    console.print(
                        f"  {prov_a} vs {prov_b}: "
                        f"Δ={comparison.mean_diff:+.3f} "
                        f"(p={comparison.p_value:.3f}{sig_marker}) "
                        f"[{winner} wins]"
                    )
