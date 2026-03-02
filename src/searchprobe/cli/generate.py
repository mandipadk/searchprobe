"""Generate adversarial queries command."""

from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from searchprobe.config import get_settings
from searchprobe.queries.generator import generate_query_set
from searchprobe.queries.taxonomy import AdversarialCategory, get_all_categories
from searchprobe.storage import Database

console = Console()
app = typer.Typer(help="Generate adversarial queries.")


@app.callback(invoke_without_command=True)
def generate(
    categories: Optional[str] = typer.Option(
        None,
        "--categories",
        "-c",
        help="Comma-separated categories (default: all). Use 'list' to see options.",
    ),
    count_per_category: int = typer.Option(
        10,
        "--count",
        "-n",
        help="Number of LLM-generated queries per category",
    ),
    tiers: str = typer.Option(
        "seed,template,llm",
        "--tiers",
        "-t",
        help="Generation tiers: seed,template,llm (comma-separated)",
    ),
    name: Optional[str] = typer.Option(
        None,
        "--name",
        help="Name for this query set",
    ),
    no_llm: bool = typer.Option(
        False,
        "--no-llm",
        help="Skip LLM generation (useful if no API key)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be generated without actually generating",
    ),
    export_seeds: bool = typer.Option(
        False,
        "--export-seeds",
        help="Export built-in seeds to data/seeds/ directory",
    ),
) -> None:
    """Generate adversarial queries across categories.

    Uses three tiers of generation:
    - seed: Human-curated queries with known ground truth
    - template: Systematic expansion from parameterized templates
    - llm: Creative generation using Claude
    """
    settings = get_settings()

    # Handle special commands
    if categories == "list":
        _show_categories()
        return

    if export_seeds:
        _export_seeds()
        return

    # Parse categories
    if categories:
        try:
            category_list = [
                AdversarialCategory(c.strip()) for c in categories.split(",")
            ]
        except ValueError as e:
            console.print(f"[red]Invalid category: {e}[/red]")
            console.print("Use --categories list to see valid options")
            raise typer.Exit(1)
    else:
        category_list = get_all_categories()

    # Parse tiers
    tier_list = [t.strip() for t in tiers.split(",")]
    valid_tiers = {"seed", "template", "llm"}
    invalid_tiers = set(tier_list) - valid_tiers
    if invalid_tiers:
        console.print(f"[red]Invalid tiers: {invalid_tiers}[/red]")
        console.print(f"Valid tiers: {valid_tiers}")
        raise typer.Exit(1)

    # Check LLM availability
    use_llm = "llm" in tier_list and not no_llm
    if use_llm and not settings.anthropic_api_key:
        console.print("[yellow]Warning: No Anthropic API key configured.[/yellow]")
        console.print("LLM generation will be skipped. Set SEARCHPROBE_ANTHROPIC_API_KEY in .env")
        use_llm = False

    # Show plan
    console.print("\n[bold]Query Generation Plan[/bold]")
    console.print(f"  Categories: {len(category_list)}")
    console.print(f"  Tiers: {tier_list}")
    console.print(f"  LLM count per category: {count_per_category}")
    console.print(f"  Use LLM: {use_llm}")
    console.print()

    if dry_run:
        console.print("[yellow]Dry run - no queries will be generated[/yellow]")
        _show_dry_run_estimate(category_list, tier_list, count_per_category, use_llm)
        return

    # Generate queries
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Generating queries...", total=None)

        query_set = generate_query_set(
            name=name,
            count_per_category=count_per_category,
            tiers=tier_list,
            categories=category_list,
            use_llm=use_llm,
        )

        progress.update(task, description="Saving to database...")

        # Save to database
        db = Database(settings.database_path)
        query_set_id = db.create_query_set(
            name=query_set.name, config=query_set.config
        )

        for query in query_set.queries:
            # Serialize ground truth properly (use mode="json" to handle datetime)
            ground_truth_dict = None
            if query.ground_truth:
                ground_truth_dict = query.ground_truth.model_dump(mode="json", exclude_none=True)

            db.add_query(
                query_set_id=query_set_id,
                text=query.text,
                category=query.category.value,
                difficulty=query.difficulty,
                tier=query.tier,
                ground_truth=ground_truth_dict,
                metadata={
                    "adversarial_reason": query.adversarial_reason,
                    **query.metadata,
                },
            )

    # Display summary
    console.print(f"\n[green]Generated {query_set.total_queries} queries[/green]")
    console.print(f"Query set ID: {query_set_id}")
    console.print()

    _show_summary(query_set)


def _show_categories() -> None:
    """Display available adversarial categories."""
    from searchprobe.queries.taxonomy import CATEGORY_METADATA

    table = Table(title="Adversarial Query Categories")
    table.add_column("Category", style="cyan")
    table.add_column("Display Name", style="green")
    table.add_column("Difficulty", style="yellow")
    table.add_column("Description", max_width=50)

    for cat, meta in CATEGORY_METADATA.items():
        table.add_row(
            cat.value,
            meta.display_name,
            meta.difficulty,
            meta.description,
        )

    console.print(table)


def _export_seeds() -> None:
    """Export built-in seeds to files."""
    from searchprobe.queries.seeds import export_builtin_seeds

    console.print("Exporting built-in seeds to data/seeds/...")
    export_builtin_seeds()
    console.print("[green]Done![/green]")


def _show_dry_run_estimate(
    categories: list[AdversarialCategory],
    tiers: list[str],
    llm_count: int,
    use_llm: bool,
) -> None:
    """Show estimated query counts for dry run."""
    from searchprobe.queries.seeds import get_builtin_seeds
    from searchprobe.queries.templates import TEMPLATES

    table = Table(title="Estimated Query Counts")
    table.add_column("Category", style="cyan")
    table.add_column("Seeds", justify="right")
    table.add_column("Templates", justify="right")
    table.add_column("LLM", justify="right")
    table.add_column("Total", justify="right", style="green")

    seeds = get_builtin_seeds()
    total_all = 0

    for cat in categories:
        seed_count = len([s for s in seeds if s.category == cat]) if "seed" in tiers else 0
        template_count = len(TEMPLATES.get(cat, [])) * 3 if "template" in tiers else 0  # ~3 per template
        llm_count_cat = llm_count if use_llm and "llm" in tiers else 0
        total = seed_count + template_count + llm_count_cat
        total_all += total

        table.add_row(
            cat.value,
            str(seed_count),
            str(template_count),
            str(llm_count_cat),
            str(total),
        )

    console.print(table)
    console.print(f"\n[bold]Estimated total: {total_all} queries[/bold]")


def _show_summary(query_set) -> None:
    """Show summary of generated queries."""
    table = Table(title="Generation Summary")
    table.add_column("Category", style="cyan")
    table.add_column("Seed", justify="right")
    table.add_column("Template", justify="right")
    table.add_column("LLM", justify="right")
    table.add_column("Total", justify="right", style="green")

    by_category = query_set.queries_by_category()

    for cat in AdversarialCategory:
        queries = by_category.get(cat, [])
        seed = len([q for q in queries if q.tier == "seed"])
        template = len([q for q in queries if q.tier == "template"])
        llm = len([q for q in queries if q.tier == "llm"])

        if seed + template + llm > 0:
            table.add_row(
                cat.value, str(seed), str(template), str(llm), str(len(queries))
            )

    console.print(table)

    # Show tier totals
    by_tier = {"seed": 0, "template": 0, "llm": 0}
    for q in query_set.queries:
        by_tier[q.tier] = by_tier.get(q.tier, 0) + 1

    console.print(f"\nBy tier: Seed={by_tier['seed']}, Template={by_tier['template']}, LLM={by_tier['llm']}")
