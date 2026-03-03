"""CLI command for cross-encoder validation."""

from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from searchprobe.config import get_settings
from searchprobe.storage import Database

console = Console()
app = typer.Typer(help="Validate search results using cross-encoder reranking.")


@app.callback(invoke_without_command=True)
def validate(
    run_id: str = typer.Option(
        "latest",
        "--run-id",
        "-r",
        help="Run ID to validate (or 'latest')",
    ),
    cross_encoder: str = typer.Option(
        "cross-encoder/ms-marco-MiniLM-L-12-v2",
        "--cross-encoder",
        "-e",
        help="Cross-encoder model name",
    ),
    max_results: Optional[int] = typer.Option(
        None,
        "--max-results",
        "-n",
        help="Maximum number of results to validate",
    ),
    categories: Optional[str] = typer.Option(
        None,
        "--categories",
        "-c",
        help="Comma-separated categories to validate",
    ),
) -> None:
    """Validate search results with cross-encoder to measure embedding gap."""
    settings = get_settings()
    db = Database(settings.database_path)

    # Resolve run ID
    actual_run_id = run_id
    if run_id == "latest":
        actual_run_id = db.get_latest_run_id()

    if not actual_run_id:
        console.print("[red]Error: No runs found.[/red]")
        raise typer.Exit(1)

    console.print(f"[blue]Validating run: {actual_run_id[:8]}...[/blue]")

    # Get search results
    search_results = db.get_search_results_for_evaluation(
        run_id=actual_run_id,
        skip_evaluated=False,
        max_results=max_results,
    )

    if categories:
        category_list = [c.strip().lower() for c in categories.split(",")]
        search_results = [r for r in search_results if r["category"] in category_list]

    if not search_results:
        console.print("[yellow]No results to validate.[/yellow]")
        raise typer.Exit(0)

    console.print(f"Found {len(search_results)} results to validate")

    # Initialize cross-encoder
    try:
        from searchprobe.validation.cross_encoder import CrossEncoderValidator

        validator = CrossEncoderValidator(model_name=cross_encoder)
    except ImportError:
        console.print("[red]Error: sentence-transformers required.[/red]")
        console.print("Install with: pip install 'searchprobe[analysis]'")
        raise typer.Exit(1)

    # Validate
    all_validations = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Validating...", total=len(search_results))

        for i, result_item in enumerate(search_results):
            progress.update(
                task,
                description=f"Validating: {result_item['query_text'][:40]}...",
            )

            results_dicts = [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": r.get("snippet", ""),
                    "content": r.get("content", ""),
                }
                for r in result_item.get("results", [])
            ]

            validation = validator.validate_search_results(
                query_id=result_item["query_id"],
                query_text=result_item["query_text"],
                category=result_item["category"],
                provider=result_item["provider"],
                results=results_dicts,
            )

            # Store in database
            db.add_validation_result({
                "run_id": actual_run_id,
                "query_id": result_item["query_id"],
                "provider": result_item["provider"],
                "cross_encoder_model": cross_encoder,
                "original_ndcg": validation.original_ndcg,
                "reranked_ndcg": validation.reranked_ndcg,
                "ndcg_improvement": validation.ndcg_improvement,
                "kendall_tau": validation.kendall_tau,
                "scores": [s.to_dict() for s in validation.scores],
            })

            all_validations.append(validation)
            progress.update(task, completed=i + 1)

    # Display summary
    console.print(f"\n[green]Validated {len(all_validations)} results[/green]")
    _display_summary(all_validations)


def _display_summary(validations: list) -> None:
    """Display validation summary."""
    if not validations:
        return

    import numpy as np

    improvements = [v.ndcg_improvement for v in validations]
    taus = [v.kendall_tau for v in validations]

    console.print("\n[bold]Embedding Gap Summary[/bold]")
    console.print(f"  Mean NDCG improvement: {np.mean(improvements):.3f} ({np.mean(improvements):.1%})")
    console.print(f"  Median NDCG improvement: {np.median(improvements):.3f}")
    console.print(f"  Mean Kendall's tau: {np.mean(taus):.3f}")

    # By category
    table = Table(title="Embedding Gap by Category")
    table.add_column("Category", style="cyan")
    table.add_column("Mean NDCG Improvement", justify="right")
    table.add_column("Mean Kendall's Tau", justify="right")
    table.add_column("N", justify="right")

    category_data: dict[str, dict[str, list[float]]] = {}
    for v in validations:
        if v.category not in category_data:
            category_data[v.category] = {"improvements": [], "taus": []}
        category_data[v.category]["improvements"].append(v.ndcg_improvement)
        category_data[v.category]["taus"].append(v.kendall_tau)

    for cat, data in sorted(category_data.items(), key=lambda x: np.mean(x[1]["improvements"]), reverse=True):
        mean_imp = np.mean(data["improvements"])
        mean_tau = np.mean(data["taus"])
        color = "red" if mean_imp > 0.2 else "yellow" if mean_imp > 0.1 else "green"
        table.add_row(
            cat.replace("_", " ").title(),
            f"[{color}]{mean_imp:.3f}[/{color}]",
            f"{mean_tau:.3f}",
            str(len(data["improvements"])),
        )

    console.print(table)
