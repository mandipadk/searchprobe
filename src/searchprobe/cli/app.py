"""Root CLI application for SearchProbe."""

import typer
from rich.console import Console

from searchprobe import __version__

# Create console for rich output
console = Console()

# Create main app
app = typer.Typer(
    name="searchprobe",
    help="Adversarial benchmark framework for neural search engines.",
    rich_markup_mode="rich",
    no_args_is_help=True,
)


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"SearchProbe version {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version",
        "-v",
        help="Show version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
) -> None:
    """SearchProbe: Adversarial benchmark framework for neural search engines."""
    pass


# Import and register subcommands
from searchprobe.cli import run as run_module  # noqa: E402
from searchprobe.cli import generate as generate_module  # noqa: E402
from searchprobe.cli import evaluate as evaluate_module  # noqa: E402
from searchprobe.cli import report as report_module  # noqa: E402
from searchprobe.cli import dashboard as dashboard_module  # noqa: E402
from searchprobe.cli import geometry as geometry_module  # noqa: E402
from searchprobe.cli import validate as validate_module  # noqa: E402
from searchprobe.cli import perturb as perturb_module  # noqa: E402
from searchprobe.cli import evolve as evolve_module  # noqa: E402

app.add_typer(run_module.app, name="run")
app.add_typer(generate_module.app, name="generate")
app.add_typer(evaluate_module.app, name="evaluate")
app.add_typer(report_module.app, name="report")
app.add_typer(dashboard_module.app, name="dashboard")
app.add_typer(geometry_module.app, name="geometry")
app.add_typer(validate_module.app, name="validate")
app.add_typer(perturb_module.app, name="perturb")
app.add_typer(evolve_module.app, name="evolve")


if __name__ == "__main__":
    app()
