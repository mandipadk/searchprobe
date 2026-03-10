"""Aris CLI -- search, serve, index, benchmark commands."""

from __future__ import annotations

import asyncio
import time

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="aris",
    help="Aris: Neural Search Engine Built on Adversarial Intelligence",
    no_args_is_help=True,
)
console = Console()


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    num_results: int = typer.Option(10, "--num", "-n", help="Number of results"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
    use_cross_encoder: bool = typer.Option(False, "--rerank", help="Enable cross-encoder reranking"),
) -> None:
    """Search using the full Aris pipeline."""
    asyncio.run(_search(query, num_results, verbose, use_cross_encoder))


async def _search(query: str, num_results: int, verbose: bool, use_cross_encoder: bool) -> None:
    from aris.agent.search_agent import SearchAgent
    from aris.core.config import ArisConfig
    from aris.index.manager import IndexManager
    from aris.que.engine import QueryUnderstandingEngine
    from aris.ranking.engine import RankingEngine
    from aris.retrieval.engine import RetrievalEngine
    from aris.sources.registry import SourceRegistry
    from aris.verification.engine import ConstraintVerificationEngine

    config = ArisConfig()

    # Initialize components
    registry = SourceRegistry(config)
    sources = registry.get_available()
    index_manager = IndexManager(config)

    que = QueryUnderstandingEngine(config)
    retrieval = RetrievalEngine(
        config,
        dense_store=index_manager.dense,
        sparse_store=index_manager.sparse,
        structured_store=index_manager.structured,
    )
    verification = ConstraintVerificationEngine()
    ranking = RankingEngine(config, use_cross_encoder=use_cross_encoder)
    agent = SearchAgent(config, que, retrieval, verification, ranking)

    # Run search
    with console.status("[bold blue]Searching..."):
        response = await agent.search(query, sources, num_results=num_results)

    # Display results
    if not response.results:
        console.print("\n[yellow]No results found.[/yellow]")
        if response.error:
            console.print(f"[red]Error: {response.error}[/red]")
        await registry.close_all()
        index_manager.close()
        return

    console.print(
        f"\n[bold green]Found {len(response.results)} results[/bold green] "
        f"in {response.latency_ms:.0f}ms ({response.iterations} iteration{'s' if response.iterations > 1 else ''})\n"
    )

    if verbose and response.predicted_failure_modes:
        console.print(f"[dim]Predicted failure modes: {', '.join(response.predicted_failure_modes)}[/dim]")
        console.print(f"[dim]Strategy: {response.strategy_used}[/dim]\n")

    table = Table(show_header=True, header_style="bold cyan", show_lines=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Title", max_width=50)
    table.add_column("Score", width=7)
    table.add_column("Confidence", width=10)
    table.add_column("Source", width=12)
    table.add_column("URL", max_width=45)

    for i, r in enumerate(response.results, 1):
        conf_style = "green" if r.confidence > 0.7 else "yellow" if r.confidence > 0.4 else "red"
        table.add_row(
            str(i),
            r.title or "[no title]",
            f"{r.score:.3f}",
            f"[{conf_style}]{r.confidence:.2f}[/{conf_style}]",
            r.source,
            r.url[:45],
        )

    console.print(table)

    if verbose:
        console.print(f"\n[dim]Total candidates evaluated: {response.total_candidates}[/dim]")
        console.print(f"[dim]Sources: {', '.join(s.name for s in sources)}[/dim]")

        for i, r in enumerate(response.results[:3], 1):
            if r.constraint_satisfaction:
                console.print(f"[dim]Result {i} constraints: {r.constraint_satisfaction}[/dim]")

    await registry.close_all()
    index_manager.close()


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to"),
) -> None:
    """Start the Aris API server."""
    try:
        import uvicorn
        from aris.api.app import create_app
        console.print(f"[bold green]Starting Aris API server on {host}:{port}[/bold green]")
        api_app = create_app()
        uvicorn.run(api_app, host=host, port=port)
    except ImportError:
        console.print("[red]Install uvicorn: pip install 'searchprobe[aris]'[/red]")
        raise typer.Exit(1)


@app.command()
def index(
    urls: list[str] = typer.Argument(..., help="URLs to fetch and index"),
) -> None:
    """Index documents from URLs into the local search index."""
    asyncio.run(_index(urls))


async def _index(urls: list[str]) -> None:
    from aris.core.config import ArisConfig
    from aris.index.manager import IndexManager
    from aris.sources.web import WebSource

    config = ArisConfig()
    web = WebSource()

    with console.status(f"[bold blue]Fetching {len(urls)} URLs..."):
        documents = await web.fetch_urls(urls)

    if not documents:
        console.print("[yellow]No documents could be fetched.[/yellow]")
        await web.close()
        return

    with console.status("[bold blue]Indexing documents..."):
        manager = IndexManager(config)
        manager.add_documents(documents)

    console.print(f"[green]Indexed {len(documents)} documents.[/green]")
    await web.close()
    manager.close()


@app.command()
def benchmark(
    categories: list[str] = typer.Option(None, "--category", "-c", help="Categories to test"),
    output: str = typer.Option(None, "--output", "-o", help="Export results to JSON file"),
) -> None:
    """Run Aris benchmark against adversarial query categories."""
    asyncio.run(_benchmark(categories, output))


async def _benchmark(categories: list[str] | None, output: str | None) -> None:
    from aris.benchmark.reporter import export_results, print_benchmark_report
    from aris.benchmark.runner import run_benchmark

    with console.status("[bold blue]Running benchmark..."):
        results = await run_benchmark(categories=categories)

    print_benchmark_report(results)

    if output:
        export_results(results, output)
        console.print(f"\n[green]Results exported to {output}[/green]")


if __name__ == "__main__":
    app()
