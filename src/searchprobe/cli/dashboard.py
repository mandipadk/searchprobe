"""Launch dashboard command."""

import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console

console = Console()
app = typer.Typer(help="Launch interactive Streamlit dashboard.")


@app.callback(invoke_without_command=True)
def dashboard(
    port: int = typer.Option(
        8501,
        "--port",
        "-p",
        help="Port to run the dashboard on",
    ),
    host: str = typer.Option(
        "localhost",
        "--host",
        "-h",
        help="Host to bind to",
    ),
    browser: bool = typer.Option(
        True,
        "--browser/--no-browser",
        help="Open browser automatically",
    ),
) -> None:
    """Launch the interactive Streamlit dashboard."""
    console.print(f"[blue]Starting SearchProbe Dashboard on http://{host}:{port}[/blue]")

    # Get the path to the dashboard app
    dashboard_path = Path(__file__).parent.parent / "dashboard" / "app.py"

    if not dashboard_path.exists():
        console.print(f"[red]Dashboard module not found at {dashboard_path}[/red]")
        raise typer.Exit(1)

    # Build streamlit command
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(dashboard_path),
        "--server.port", str(port),
        "--server.address", host,
    ]

    if not browser:
        cmd.extend(["--server.headless", "true"])

    try:
        console.print("[green]Dashboard starting...[/green]")
        console.print("Press Ctrl+C to stop\n")
        subprocess.run(cmd)
    except KeyboardInterrupt:
        console.print("\n[yellow]Dashboard stopped[/yellow]")
    except FileNotFoundError:
        console.print("[red]Streamlit not found. Install with: pip install streamlit[/red]")
        raise typer.Exit(1)
