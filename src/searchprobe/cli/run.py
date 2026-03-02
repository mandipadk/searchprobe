"""Run benchmark command."""

import asyncio
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from searchprobe.config import get_settings
from searchprobe.pipeline.runner import BenchmarkConfig, BenchmarkRunner
from searchprobe.providers.registry import ProviderRegistry
from searchprobe.storage import Database

console = Console()
app = typer.Typer(help="Run search benchmarks against providers.")


@app.callback(invoke_without_command=True)
def run(
    providers: Optional[str] = typer.Option(
        None,
        "--providers",
        "-p",
        help="Comma-separated providers to test (default: all configured). Use 'list' to see options.",
    ),
    exa_modes: str = typer.Option(
        "auto",
        "--exa-modes",
        "-m",
        help="Comma-separated Exa search modes: auto,neural,fast,deep",
    ),
    query: Optional[str] = typer.Option(
        None,
        "--query",
        "-q",
        help="Single query to run (for testing)",
    ),
    query_set: str = typer.Option(
        "latest",
        "--query-set",
        help="Query set ID to use (or 'latest')",
    ),
    max_queries: Optional[int] = typer.Option(
        None,
        "--max-queries",
        "-n",
        help="Maximum number of queries to run",
    ),
    num_results: int = typer.Option(
        10,
        "--num-results",
        "-r",
        help="Number of results per query",
    ),
    run_name: Optional[str] = typer.Option(
        None,
        "--name",
        help="Human-readable name for this run",
    ),
    budget: float = typer.Option(
        10.0,
        "--budget",
        "-b",
        help="Maximum budget in USD",
    ),
) -> None:
    """Run search benchmark against specified providers."""
    settings = get_settings()

    # Handle list command
    if providers == "list":
        _show_providers()
        return

    # Validate and get provider list
    if providers:
        provider_list = [p.strip().lower() for p in providers.split(",")]
        # Validate providers exist
        available = ProviderRegistry.list_available()
        invalid = [p for p in provider_list if p not in available]
        if invalid:
            console.print(f"[red]Unknown providers: {invalid}[/red]")
            console.print(f"Available: {available}")
            raise typer.Exit(1)
    else:
        provider_list = ProviderRegistry.list_configured(settings)

    if not provider_list:
        console.print("[red]Error: No providers configured.[/red]")
        console.print("Set API keys in .env file. Example:")
        console.print("  SEARCHPROBE_EXA_API_KEY=your-key")
        console.print("\nAvailable providers:")
        for name in ProviderRegistry.list_available():
            info = ProviderRegistry.get_provider_info(name)
            console.print(f"  {name}: modes={info['supported_modes']}")
        raise typer.Exit(1)

    # Check configured providers
    configured = ProviderRegistry.list_configured(settings)
    unconfigured = [p for p in provider_list if p not in configured]
    if unconfigured:
        console.print(f"[red]Error: Providers not configured: {unconfigured}[/red]")
        console.print("Set the API keys in .env file")
        raise typer.Exit(1)

    # Parse Exa modes
    modes = [m.strip() for m in exa_modes.split(",")]

    # Get queries
    db = Database(settings.database_path)
    queries_to_run: list[dict] = []

    if query:
        # Single query mode (for testing)
        queries_to_run = [{"id": "test", "text": query, "category": "test"}]
        console.print(f"[blue]Running single query: {query}[/blue]")
    else:
        # Load from database
        query_set_id = query_set
        if query_set == "latest":
            query_set_id = db.get_latest_query_set_id()

        if not query_set_id:
            console.print("[yellow]No query sets found. Running with sample query.[/yellow]")
            queries_to_run = [
                {
                    "id": "sample",
                    "text": "companies that are NOT in AI",
                    "category": "negation",
                }
            ]
        else:
            queries_to_run = db.get_queries(query_set_id=query_set_id, limit=max_queries)
            console.print(f"[blue]Loaded {len(queries_to_run)} queries from set {query_set_id[:8]}...[/blue]")

    if not queries_to_run:
        console.print("[red]No queries to run.[/red]")
        raise typer.Exit(1)

    # Apply max_queries limit
    if max_queries:
        queries_to_run = queries_to_run[:max_queries]

    # Calculate estimated tasks
    num_tasks = len(queries_to_run) * (len(modes) if "exa" in provider_list else 0)
    num_tasks += len(queries_to_run) * len([p for p in provider_list if p != "exa"])

    # Show plan
    console.print(f"\n[bold]Running benchmark[/bold]")
    console.print(f"  Providers: {', '.join(provider_list)}")
    if "exa" in provider_list:
        console.print(f"  Exa modes: {', '.join(modes)}")
    console.print(f"  Queries: {len(queries_to_run)}")
    console.print(f"  Total tasks: {num_tasks}")
    console.print(f"  Budget: ${budget:.2f}")
    console.print()

    # Create config
    config = BenchmarkConfig(
        providers=provider_list,
        exa_modes=modes,
        num_results=num_results,
        budget_limit=budget,
    )

    # Execute with progress
    runner = BenchmarkRunner(config, settings, db)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Initializing...", total=num_tasks)

        def update_progress(status: str, completed: int, total: int):
            progress.update(task, description=status, completed=completed, total=total)

        runner.set_progress_callback(update_progress)

        result = asyncio.run(runner.run(queries_to_run, run_name=run_name))

    # Display results
    console.print(f"\n[green]Run completed: {result.run_id}[/green]")
    console.print(f"  Duration: {(result.completed_at - result.started_at).total_seconds():.1f}s")
    console.print(f"  Successful: {result.successful_queries}")
    console.print(f"  Failed: {result.failed_queries}")
    console.print(f"  Total cost: ${result.total_cost:.4f}")

    if result.cost_by_provider:
        console.print("  By provider:")
        for provider, cost in result.cost_by_provider.items():
            console.print(f"    {provider}: ${cost:.4f}")

    # Show results table
    _display_results(result.results)


def _show_providers() -> None:
    """Display available and configured providers."""
    settings = get_settings()
    configured = set(ProviderRegistry.list_configured(settings))

    table = Table(title="Search Providers")
    table.add_column("Provider", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Modes", style="blue")
    table.add_column("Cost/Query", justify="right")

    for name in ProviderRegistry.list_available():
        info = ProviderRegistry.get_provider_info(name)
        status = "[green]Configured[/green]" if name in configured else "[yellow]Not configured[/yellow]"
        modes = ", ".join(info["supported_modes"])
        costs = ", ".join(f"${v}" for v in info["cost_per_query"].values())

        table.add_row(name, status, modes, costs)

    console.print(table)


def _display_results(results) -> None:
    """Display results in a rich table."""
    if not results:
        console.print("[yellow]No results to display.[/yellow]")
        return

    # Group by provider for summary
    by_provider: dict[str, list] = {}
    for r in results:
        provider_key = f"{r.provider}:{r.search_mode or 'default'}"
        if provider_key not in by_provider:
            by_provider[provider_key] = []
        by_provider[provider_key].append(r)

    # Summary table
    table = Table(title="Results by Provider")
    table.add_column("Provider", style="cyan")
    table.add_column("Mode", style="blue")
    table.add_column("Queries", justify="right")
    table.add_column("Success", justify="right", style="green")
    table.add_column("Failed", justify="right", style="red")
    table.add_column("Avg Latency", justify="right")
    table.add_column("Total Cost", justify="right")

    for key, provider_results in by_provider.items():
        parts = key.split(":", 1)
        provider = parts[0]
        mode = parts[1] if len(parts) > 1 else "-"

        total = len(provider_results)
        success = len([r for r in provider_results if r.success])
        failed = len([r for r in provider_results if r.error])
        avg_latency = sum(r.latency_ms for r in provider_results) / total if total > 0 else 0
        total_cost = sum(r.cost_usd for r in provider_results)

        table.add_row(
            provider,
            mode,
            str(total),
            str(success),
            str(failed),
            f"{avg_latency:.0f}ms",
            f"${total_cost:.4f}",
        )

    console.print(table)

    # Show a few example results
    console.print("\n[bold]Sample Results:[/bold]")
    sample_results = results[:5]

    for r in sample_results:
        status = "[green]OK[/green]" if r.success else f"[red]ERR: {r.error}[/red]"
        console.print(f"  [{r.provider}:{r.search_mode or 'default'}] {r.query[:50]}... -> {status}")
        if r.results:
            console.print(f"    Top: {r.results[0].title[:60]}...")
