"""CLI command for adversarial query evolution."""

import asyncio
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from searchprobe.config import get_settings

console = Console()
app = typer.Typer(help="Evolve adversarial queries using evolutionary optimization.")


@app.callback(invoke_without_command=True)
def evolve(
    provider: str = typer.Option(
        "exa",
        "--provider",
        "-p",
        help="Provider to test against",
    ),
    mode: Optional[str] = typer.Option(
        None,
        "--mode",
        help="Search mode (e.g., 'neural', 'auto')",
    ),
    generations: int = typer.Option(
        20,
        "--generations",
        "-g",
        help="Number of evolution generations",
    ),
    population: int = typer.Option(
        30,
        "--population",
        "-n",
        help="Population size",
    ),
    budget: float = typer.Option(
        5.0,
        "--budget",
        "-b",
        help="Maximum budget in USD",
    ),
    fitness_mode: str = typer.Option(
        "embedding_sim",
        "--fitness",
        "-f",
        help="Fitness mode: embedding_sim, llm_judge, cross_encoder",
    ),
    categories: Optional[str] = typer.Option(
        None,
        "--categories",
        "-c",
        help="Target categories (comma-separated)",
    ),
    seeds: Optional[str] = typer.Option(
        None,
        "--seeds",
        "-s",
        help="Seed queries (comma-separated)",
    ),
) -> None:
    """Evolve adversarial queries to discover worst-case failure modes."""
    from searchprobe.adversarial.fitness import FitnessEvaluator
    from searchprobe.adversarial.models import EvolutionConfig
    from searchprobe.adversarial.optimizer import AdversarialQueryOptimizer

    # Parse categories
    target_categories = []
    if categories:
        target_categories = [c.strip() for c in categories.split(",")]

    # Parse or generate seed queries
    seed_queries = []
    if seeds:
        seed_queries = [s.strip() for s in seeds.split(",")]
    else:
        # Default seeds from taxonomy
        from searchprobe.queries.taxonomy import get_example_queries

        seed_queries = get_example_queries()[:population]

    console.print(f"[blue]Evolving adversarial queries...[/blue]")
    console.print(f"  Population: {population}")
    console.print(f"  Generations: {generations}")
    console.print(f"  Fitness mode: {fitness_mode}")
    console.print(f"  Seed queries: {len(seed_queries)}")

    # Configure evolution
    config = EvolutionConfig(
        population_size=population,
        generations=generations,
        budget_limit=budget,
        fitness_mode=fitness_mode,
        seed_queries=seed_queries,
        target_categories=target_categories,
    )

    # Initialize fitness evaluator
    search_provider = None
    judge = None

    if fitness_mode in ("llm_judge", "cross_encoder"):
        settings = get_settings()
        from searchprobe.providers.registry import ProviderRegistry

        try:
            search_provider = ProviderRegistry.get_provider(provider, settings)
        except Exception as e:
            console.print(f"[red]Error: Cannot initialize provider '{provider}': {e}[/red]")
            raise typer.Exit(1)

        if fitness_mode == "llm_judge":
            from searchprobe.evaluation.judge import SearchJudge

            judge = SearchJudge(settings)

    fitness_evaluator = FitnessEvaluator(
        mode=fitness_mode,
        provider=search_provider,
        judge=judge,
    )

    # Run evolution
    optimizer = AdversarialQueryOptimizer(config, fitness_evaluator)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Evolving...", total=generations)

        def on_progress(gen: int, total: int, best_fitness: float, mean_fitness: float) -> None:
            progress.update(
                task,
                description=f"Gen {gen}/{total} | Best: {best_fitness:.3f} | Mean: {mean_fitness:.3f}",
                completed=gen,
            )

        result = asyncio.run(optimizer.optimize(progress_callback=on_progress))

    # Persist results to database
    from searchprobe.storage import Database

    settings = get_settings()
    db = Database(settings.database_path)
    db.add_evolution_result({
        "fitness_mode": fitness_mode,
        "provider": provider,
        "generations_completed": result.generations_completed,
        "total_evaluations": result.total_evaluations,
        "total_cost": result.total_cost,
        "best_individuals": [i.to_dict() for i in result.best_individuals],
        "fitness_history": result.fitness_history,
        "config": {
            "population_size": population,
            "generations": generations,
            "budget_limit": budget,
            "fitness_mode": fitness_mode,
        },
    })

    # Display results
    console.print(f"\n[green]Evolution complete![/green]")
    console.print(f"  Generations: {result.generations_completed}")
    console.print(f"  Total evaluations: {result.total_evaluations}")

    # Top adversarial queries
    table = Table(title="Top Evolved Adversarial Queries")
    table.add_column("#", style="dim", width=3)
    table.add_column("Query", style="cyan", max_width=60)
    table.add_column("Fitness", justify="right")
    table.add_column("Category", style="blue")
    table.add_column("Mutations", justify="right")

    for i, ind in enumerate(result.best_individuals[:10]):
        table.add_row(
            str(i + 1),
            ind.query[:60],
            f"{ind.fitness:.3f}",
            ind.category or "-",
            str(len(ind.mutation_history)),
        )

    console.print(table)

    # Fitness history
    if result.fitness_history:
        console.print("\n[bold]Fitness History[/bold]")
        first = result.fitness_history[0]
        last = result.fitness_history[-1]
        console.print(f"  Gen 0:  mean={first['mean']:.3f}, max={first['max']:.3f}")
        console.print(f"  Gen {result.generations_completed}: mean={last['mean']:.3f}, max={last['max']:.3f}")
        improvement = last["max"] - first["max"]
        console.print(f"  Improvement: +{improvement:.3f}")

    # Show mutation history for best query
    best = result.best_individuals[0] if result.best_individuals else None
    if best and best.mutation_history:
        console.print(f"\n[bold]Best Query Mutation History[/bold]")
        for j, mutation in enumerate(best.mutation_history):
            console.print(f"  {j + 1}. {mutation}")
