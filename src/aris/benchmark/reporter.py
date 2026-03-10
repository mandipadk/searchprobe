"""Generates comparative analysis reports for Aris vs other providers."""

from __future__ import annotations

import json
from typing import Any

from rich.console import Console
from rich.table import Table


def print_benchmark_report(results: dict[str, Any]) -> None:
    """Print a formatted benchmark comparison report."""
    console = Console()

    console.print("\n[bold cyan]Aris Benchmark Report[/bold cyan]\n")

    for category, provider_results in results.items():
        console.print(f"\n[bold]{category.replace('_', ' ').title()}[/bold]")

        table = Table(show_header=True, header_style="bold")
        table.add_column("Provider", width=15)
        table.add_column("Queries", width=10)
        table.add_column("Success Rate", width=15)
        table.add_column("Avg Results", width=12)
        table.add_column("Avg Latency", width=12)

        for provider_name, query_results in provider_results.items():
            if not query_results:
                continue

            total = len(query_results)
            successes = sum(1 for r in query_results if r["success"])
            avg_results = sum(r["num_results"] for r in query_results) / total
            avg_latency = sum(r["latency_ms"] for r in query_results) / total

            success_rate = f"{successes}/{total} ({successes/total:.0%})"
            style = "green" if successes == total else "yellow" if successes > 0 else "red"

            table.add_row(
                provider_name,
                str(total),
                f"[{style}]{success_rate}[/{style}]",
                f"{avg_results:.1f}",
                f"{avg_latency:.0f}ms",
            )

        console.print(table)

    # Summary
    console.print("\n[bold]Overall Summary[/bold]")
    summary_table = Table(show_header=True, header_style="bold")
    summary_table.add_column("Provider", width=15)
    summary_table.add_column("Total Queries", width=15)
    summary_table.add_column("Overall Success", width=15)

    all_providers: dict[str, dict[str, int]] = {}
    for category, provider_results in results.items():
        for provider_name, query_results in provider_results.items():
            if provider_name not in all_providers:
                all_providers[provider_name] = {"total": 0, "success": 0}
            all_providers[provider_name]["total"] += len(query_results)
            all_providers[provider_name]["success"] += sum(
                1 for r in query_results if r["success"]
            )

    for provider_name, stats in all_providers.items():
        rate = stats["success"] / stats["total"] if stats["total"] > 0 else 0
        style = "green" if rate > 0.8 else "yellow" if rate > 0.5 else "red"
        summary_table.add_row(
            provider_name,
            str(stats["total"]),
            f"[{style}]{rate:.0%}[/{style}]",
        )

    console.print(summary_table)


def export_results(results: dict[str, Any], path: str) -> None:
    """Export benchmark results to JSON."""
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
