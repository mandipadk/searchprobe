"""CLI command for embedding geometry analysis."""

from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()
app = typer.Typer(help="Analyze embedding geometry to understand why search fails.")


@app.callback(invoke_without_command=True)
def geometry(
    models: str = typer.Option(
        "all-MiniLM-L6-v2",
        "--models",
        "-m",
        help="Comma-separated embedding model names",
    ),
    categories: Optional[str] = typer.Option(
        None,
        "--categories",
        "-c",
        help="Comma-separated categories (default: all)",
    ),
    format: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="Output format: table, json",
    ),
    save: bool = typer.Option(
        True,
        "--save/--no-save",
        help="Save results to database",
    ),
) -> None:
    """Analyze embedding space geometry for adversarial categories."""
    from searchprobe.geometry.analyzer import EmbeddingGeometryAnalyzer
    from searchprobe.geometry.vulnerability import classify_vulnerability

    model_list = [m.strip() for m in models.split(",")]
    category_list = [c.strip() for c in categories.split(",")] if categories else None

    console.print(f"[blue]Analyzing embedding geometry with {len(model_list)} model(s)...[/blue]")

    try:
        analyzer = EmbeddingGeometryAnalyzer(models=model_list)
    except ImportError:
        console.print(
            "[red]Error: sentence-transformers is required for geometry analysis.[/red]"
        )
        console.print("Install with: pip install 'searchprobe[analysis]'")
        raise typer.Exit(1)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Analyzing...", total=None)

        def on_progress(model: str, category: str, model_idx: int, cat_idx: int, total: int) -> None:
            progress.update(task, description=f"Model {model_idx + 1}/{total}: {model} - {category}")

        report = analyzer.generate_report(
            categories=category_list,
            progress_callback=on_progress,
        )

    # Display results
    if format == "json":
        import json
        data = {}
        for model, cats in report.profiles.items():
            data[model] = {cat: prof.to_dict() for cat, prof in cats.items()}
        console.print_json(json.dumps(data, indent=2))
    else:
        _display_table(report, classify_vulnerability)

    # Save to database
    if save:
        from searchprobe.config import get_settings
        from searchprobe.storage import Database

        settings = get_settings()
        db = Database(settings.database_path)

        for model, cats in report.profiles.items():
            for cat, profile in cats.items():
                db.add_geometry_result(profile.to_dict())

        console.print(f"[green]Results saved to database[/green]")


def _display_table(report: object, classify: object) -> None:
    """Display geometry report as a rich table."""
    table = Table(title="Embedding Vulnerability Analysis")
    table.add_column("Category", style="cyan")

    for model in report.models:
        table.add_column(model, justify="center")

    vulnerability_matrix = report.get_vulnerability_matrix()

    # Get all categories
    all_categories = set()
    for model_data in vulnerability_matrix.values():
        all_categories.update(model_data.keys())

    for category in sorted(all_categories):
        row = [category.replace("_", " ").title()]
        for model in report.models:
            score = vulnerability_matrix.get(model, {}).get(category, 0)
            severity = classify(score)
            if severity == "critical":
                row.append(f"[bold red]{score:.2f}[/bold red]")
            elif severity == "high":
                row.append(f"[red]{score:.2f}[/red]")
            elif severity == "moderate":
                row.append(f"[yellow]{score:.2f}[/yellow]")
            else:
                row.append(f"[green]{score:.2f}[/green]")
        table.add_row(*row)

    console.print(table)

    # Summary
    for model in report.models:
        top_vulns = report.get_most_vulnerable_categories(model, top_n=3)
        console.print(f"\n[bold]{model}[/bold] — Most vulnerable:")
        for cat, score in top_vulns:
            console.print(f"  {cat.replace('_', ' ').title()}: {score:.3f}")
