"""Run a full research pipeline from a TOML profile."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console

from searchprobe.config import get_settings

console = Console()
app = typer.Typer(help="cases -> search -> judge -> scorecard -> metamorphic, from a profile.")


@app.callback(invoke_without_command=True)
def research(
    profile: Path = typer.Option(..., "--profile", "-P", help="TOML research profile"),
) -> None:
    """Run every stage of a research profile and write a report directory."""
    from searchprobe.pipeline.research import ResearchProfile, run_research

    settings = get_settings()
    prof = ResearchProfile.from_toml(profile)
    if prof.judge and not settings.has_anthropic_configured():
        console.print("[red]Judging needs Anthropic credentials (or set judge = false).[/red]")
        raise typer.Exit(1)
    console.print(f"[bold]Research run: {prof.name}[/bold] systems={', '.join(prof.systems)}")
    summary = asyncio.run(run_research(prof, settings, log=console.print))
    console.print(f"[green]Reports in {summary['out_dir']}[/green]")
