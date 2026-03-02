"""Generate benchmark reports command."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from searchprobe.config import get_settings
from searchprobe.reporting.generator import ReportGenerator
from searchprobe.storage import Database

console = Console()
app = typer.Typer(help="Generate benchmark reports.")


@app.callback(invoke_without_command=True)
def report(
    run_id: str = typer.Option(
        "latest",
        "--run-id",
        "-r",
        help="Run ID to generate report for (or 'latest')",
    ),
    format: str = typer.Option(
        "both",
        "--format",
        "-f",
        help="Output format: markdown, html, or both",
    ),
    output_dir: str = typer.Option(
        "reports",
        "--output-dir",
        "-o",
        help="Output directory for reports",
    ),
    open_browser: bool = typer.Option(
        False,
        "--open/--no-open",
        help="Open HTML report in browser after generation",
    ),
) -> None:
    """Generate benchmark reports with charts and analysis."""
    settings = get_settings()
    db = Database(settings.database_path)

    # Resolve run ID
    actual_run_id = run_id
    if run_id == "latest":
        actual_run_id = db.get_latest_run_id()

    if not actual_run_id:
        console.print("[red]Error: No runs found to report on.[/red]")
        console.print("Run 'searchprobe run' first, then 'searchprobe evaluate'.")
        raise typer.Exit(1)

    # Check for evaluations
    evaluations = db.get_evaluations(actual_run_id)
    if not evaluations:
        console.print("[yellow]Warning: No evaluations found for this run.[/yellow]")
        console.print("Run 'searchprobe evaluate' first to generate quality scores.")
        console.print("Generating report with available data...")

    console.print(f"[blue]Generating report for run: {actual_run_id[:8]}...[/blue]")

    # Validate format
    valid_formats = ("markdown", "html", "both")
    if format.lower() not in valid_formats:
        console.print(f"[red]Invalid format: {format}[/red]")
        console.print(f"Valid formats: {valid_formats}")
        raise typer.Exit(1)

    # Generate reports
    generator = ReportGenerator(settings, output_dir=output_dir)

    try:
        output_files = generator.generate(
            run_id=actual_run_id,
            format=format.lower(),
            db=db,
        )

        console.print("\n[green]Reports generated successfully![/green]")
        for fmt, path in output_files.items():
            console.print(f"  {fmt.capitalize()}: {path}")

        # Open in browser if requested
        if open_browser and "html" in output_files:
            import webbrowser
            html_path = output_files["html"]
            url = f"file://{html_path.absolute()}"
            console.print(f"\n[blue]Opening in browser: {url}[/blue]")
            webbrowser.open(url)

    except Exception as e:
        console.print(f"[red]Error generating report: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def list_runs() -> None:
    """List available benchmark runs."""
    settings = get_settings()
    db = Database(settings.database_path)

    with db._get_connection() as conn:
        rows = conn.execute(
            """SELECT id, name, started_at, completed_at, cost_total
               FROM runs ORDER BY started_at DESC LIMIT 20"""
        ).fetchall()

    if not rows:
        console.print("[yellow]No benchmark runs found.[/yellow]")
        return

    from rich.table import Table

    table = Table(title="Benchmark Runs")
    table.add_column("Run ID", style="cyan")
    table.add_column("Name")
    table.add_column("Started")
    table.add_column("Cost", justify="right")

    for row in rows:
        run_id = row["id"][:8] + "..."
        name = row["name"] or "-"
        started = row["started_at"][:19] if row["started_at"] else "-"
        cost = f"${row['cost_total']:.4f}" if row["cost_total"] else "-"

        table.add_row(run_id, name, started, cost)

    console.print(table)
