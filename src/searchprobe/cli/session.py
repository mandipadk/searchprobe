"""CLI command for running research sessions from TOML profiles."""

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from searchprobe.config import get_settings

console = Console()
app = typer.Typer(help="Run a research session from a TOML profile.")


@app.callback(invoke_without_command=True)
def session(
    profile: str = typer.Option(
        ...,
        "--profile",
        "-P",
        help="Path to a TOML profile file",
    ),
) -> None:
    """Run a full research session defined by a TOML profile."""
    from searchprobe.core.config import SearchProbeProfile
    from searchprobe.intelligence.session import ResearchSession

    # Load profile
    profile_path = Path(profile)
    if not profile_path.exists():
        console.print(f"[red]Profile not found: {profile_path}[/red]")
        raise typer.Exit(1)

    try:
        prof = SearchProbeProfile.from_toml(profile_path)
    except Exception as e:
        console.print(f"[red]Error loading profile: {e}[/red]")
        raise typer.Exit(1)

    console.print(f"[blue]Research Session: {prof.name}[/blue]")
    if prof.description:
        console.print(f"  {prof.description}")
    console.print(f"  Providers: {', '.join(prof.providers)}")
    console.print(f"  Categories: {', '.join(prof.categories) or 'all'}")
    console.print(f"  Analyses: ", end="")
    analyses = []
    if prof.run_geometry:
        analyses.append("geometry")
    if prof.run_perturbation:
        analyses.append("perturbation")
    if prof.run_validation:
        analyses.append("validation")
    if prof.run_evolution:
        analyses.append("evolution")
    console.print(", ".join(analyses) or "none")

    # Build session
    settings = get_settings()
    research_session = ResearchSession.from_profile(prof, settings)

    if not research_session.stages:
        console.print("[yellow]No analysis stages enabled in profile.[/yellow]")
        raise typer.Exit(0)

    console.print(f"  Stages: {', '.join(research_session.stages.keys())}")
    console.print()

    # Run session
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Starting...", total=len(research_session.stages))

        def on_progress(stage: str, completed: int, total: int) -> None:
            progress.update(
                task,
                description=f"Stage: {stage}",
                completed=completed,
            )

        context = asyncio.run(research_session.run(progress_callback=on_progress))

    # Display results
    console.print("\n[green]Session complete![/green]\n")

    # Summary table
    summary_table = Table(title="Stage Results")
    summary_table.add_column("Stage", style="cyan")
    summary_table.add_column("Categories", justify="right")
    summary_table.add_column("Signals", justify="right")
    summary_table.add_column("Key Metric", style="yellow")

    for name, result in context.results.items():
        key_metric = ""
        if name == "geometry":
            key_metric = f"Mean vulnerability: {result.summary.get('mean_vulnerability', 'N/A')}"
        elif name == "perturbation":
            key_metric = f"Mean Jaccard: {result.summary.get('mean_jaccard', 'N/A')}"
        elif name == "validation":
            key_metric = f"Mean NDCG improvement: {result.summary.get('mean_ndcg_improvement', 'N/A')}"
        elif name == "evolution":
            key_metric = f"Best fitness: {result.summary.get('best_fitness', 'N/A')}"
        elif name == "correlation":
            key_metric = f"High risk: {result.summary.get('high_risk_categories', 0)}"

        summary_table.add_row(
            name,
            str(len(result.categories)),
            str(len(result.signals)),
            key_metric,
        )

    console.print(summary_table)

    # Vulnerable categories
    if context.vulnerable_categories:
        console.print(f"\n[bold red]Vulnerable Categories:[/bold red] {', '.join(context.vulnerable_categories)}")

    # Correlation profiles (if available)
    if "correlation" in context.results:
        corr_result = context.results["correlation"]
        if corr_result.details:
            console.print("\n[bold]Category Intelligence Profiles[/bold]")
            profile_table = Table()
            profile_table.add_column("Category", style="cyan")
            profile_table.add_column("Risk", justify="center")
            profile_table.add_column("Vulnerability", justify="right")
            profile_table.add_column("Stability", justify="right")
            profile_table.add_column("Gap", justify="right")
            profile_table.add_column("Top Recommendation")

            for detail in corr_result.details[:10]:
                sv = detail.get("signal_vector", {})
                recs = detail.get("recommendations", [])
                risk_level = detail.get("risk_level", "?")
                risk_style = {"HIGH": "red", "MEDIUM": "yellow", "LOW": "green"}.get(risk_level, "")

                profile_table.add_row(
                    detail.get("category", "?"),
                    f"[{risk_style}]{risk_level}[/{risk_style}]" if risk_style else risk_level,
                    f"{sv.get('vulnerability', 'N/A')}",
                    f"{sv.get('stability', 'N/A')}",
                    f"{sv.get('embedding_gap', 'N/A')}",
                    recs[0][:80] + "..." if recs and len(recs[0]) > 80 else (recs[0] if recs else ""),
                )

            console.print(profile_table)
