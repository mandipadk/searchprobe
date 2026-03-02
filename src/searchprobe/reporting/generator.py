"""Report generator for benchmark results."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, PackageLoader, select_autoescape

from searchprobe.config import Settings, get_settings
from searchprobe.evaluation.statistics import (
    summary_statistics,
    aggregate_by_category,
    aggregate_by_provider,
    failure_mode_frequency,
)
from searchprobe.reporting.charts import (
    create_radar_chart,
    create_heatmap,
    create_bar_chart,
    create_failure_mode_chart,
    create_cost_breakdown,
)
from searchprobe.storage import Database


class ReportGenerator:
    """Generate benchmark reports in multiple formats.

    Supports Markdown and HTML output with embedded Plotly charts.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        output_dir: str | Path = "reports",
    ) -> None:
        """Initialize the report generator.

        Args:
            settings: Application settings
            output_dir: Directory for output files
        """
        self.settings = settings or get_settings()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Jinja2 environment
        try:
            self.env = Environment(
                loader=PackageLoader("searchprobe.reporting", "templates"),
                autoescape=select_autoescape(["html", "xml"]),
            )
        except Exception:
            # Fallback if templates not found
            self.env = None

    def generate(
        self,
        run_id: str,
        format: str = "both",
        db: Database | None = None,
    ) -> dict[str, Path]:
        """Generate reports for a benchmark run.

        Args:
            run_id: Run ID to generate report for
            format: Output format ('markdown', 'html', or 'both')
            db: Database instance

        Returns:
            Dict mapping format to output file path
        """
        db = db or Database(self.settings.database_path)

        # Gather data
        report_data = self._gather_data(run_id, db)

        output_files = {}

        if format in ("markdown", "both"):
            md_path = self._generate_markdown(report_data)
            output_files["markdown"] = md_path

        if format in ("html", "both"):
            html_path = self._generate_html(report_data)
            output_files["html"] = html_path

        return output_files

    def _gather_data(self, run_id: str, db: Database) -> dict[str, Any]:
        """Gather all data needed for the report.

        Args:
            run_id: Run ID
            db: Database instance

        Returns:
            Dict with all report data
        """
        # Get run info
        run_stats = db.get_run_stats(run_id)

        # Get evaluations
        evaluations = db.get_evaluations(run_id)

        # Calculate statistics
        stats = summary_statistics([{
            "category": e.get("category"),
            "provider": e.get("provider"),
            "search_mode": e.get("search_mode"),
            "weighted_score": e.get("weighted_score", 0),
            "failure_modes": e.get("failure_modes", []),
        } for e in evaluations])

        # Build provider -> category -> score mapping for charts
        by_category = aggregate_by_category([{
            "category": e.get("category"),
            "provider": e.get("provider"),
            "search_mode": e.get("search_mode"),
            "weighted_score": e.get("weighted_score", 0),
        } for e in evaluations])

        # Reformat for charts
        chart_data: dict[str, dict[str, float]] = {}
        for category, providers in by_category.items():
            for provider_key, ci in providers.items():
                if provider_key not in chart_data:
                    chart_data[provider_key] = {}
                chart_data[provider_key][category] = ci.mean

        # Get failure examples
        failure_examples = self._get_failure_examples(evaluations)

        # Build cost data
        cost_data = {}
        if run_stats.get("providers"):
            for p in run_stats["providers"]:
                cost_data[p["name"]] = p.get("cost", 0) or 0

        return {
            "run_id": run_id,
            "run_stats": run_stats,
            "evaluations": evaluations,
            "stats": stats,
            "chart_data": chart_data,
            "failure_examples": failure_examples,
            "cost_data": cost_data,
            "generated_at": datetime.utcnow().isoformat(),
        }

    def _get_failure_examples(
        self,
        evaluations: list[dict],
        max_per_category: int = 3,
    ) -> dict[str, list[dict]]:
        """Get example failures by category.

        Args:
            evaluations: List of evaluation dicts
            max_per_category: Maximum examples per category

        Returns:
            Dict of category -> list of failure examples
        """
        failures: dict[str, list[dict]] = {}

        for e in evaluations:
            if e.get("failure_modes") and e["failure_modes"]:
                category = e.get("category", "unknown")
                if category not in failures:
                    failures[category] = []
                if len(failures[category]) < max_per_category:
                    failures[category].append({
                        "query": e.get("query_text", ""),
                        "provider": e.get("provider", ""),
                        "score": e.get("weighted_score", 0),
                        "failure_modes": e.get("failure_modes", []),
                        "assessment": e.get("reasoning", ""),
                    })

        return failures

    def _generate_markdown(self, data: dict[str, Any]) -> Path:
        """Generate Markdown report.

        Args:
            data: Report data

        Returns:
            Path to output file
        """
        run_id = data["run_id"]
        stats = data["stats"]
        run_stats = data.get("run_stats", {})

        lines = [
            f"# SearchProbe Benchmark Report",
            "",
            f"**Run ID:** `{run_id[:8]}...`",
            f"**Generated:** {data['generated_at']}",
            "",
        ]

        # Run info
        if run_stats:
            lines.extend([
                "## Run Summary",
                "",
                f"- **Started:** {run_stats.get('started_at', 'N/A')}",
                f"- **Completed:** {run_stats.get('completed_at', 'N/A')}",
                f"- **Total Cost:** ${run_stats.get('cost_total', 0):.4f}",
                "",
            ])

        # Overall statistics
        lines.extend([
            "## Overall Statistics",
            "",
            f"- **Total Evaluated:** {stats.get('count', 0)}",
            f"- **Mean Score:** {stats.get('mean', 0):.3f}",
            f"- **Median Score:** {stats.get('median', 0):.3f}",
            f"- **Std Deviation:** {stats.get('std', 0):.3f}",
            "",
        ])

        # Provider summary
        if stats.get("by_provider"):
            lines.extend([
                "## Scores by Provider",
                "",
                "| Provider | Mean | 95% CI | n |",
                "|----------|------|--------|---|",
            ])
            for provider, ci in stats["by_provider"].items():
                lines.append(
                    f"| {provider} | {ci.mean:.3f} | [{ci.lower:.3f}, {ci.upper:.3f}] | {ci.n} |"
                )
            lines.append("")

        # Category summary
        if stats.get("by_category"):
            lines.extend([
                "## Scores by Category",
                "",
            ])

            # Build table header
            all_providers = set()
            for cat_data in stats["by_category"].values():
                all_providers.update(cat_data.keys())
            providers = sorted(all_providers)

            header = "| Category | " + " | ".join(providers) + " |"
            separator = "|----------|" + "|".join(["------"] * len(providers)) + "|"
            lines.extend([header, separator])

            for category, provider_scores in stats["by_category"].items():
                row = f"| {category} |"
                for provider in providers:
                    if provider in provider_scores:
                        row += f" {provider_scores[provider].mean:.2f} |"
                    else:
                        row += " - |"
                lines.append(row)
            lines.append("")

        # Failure modes
        if stats.get("failure_modes"):
            lines.extend([
                "## Top Failure Modes",
                "",
            ])
            for mode, count in list(stats["failure_modes"].items())[:10]:
                lines.append(f"- **{mode}:** {count}")
            lines.append("")

        # Failure examples
        if data.get("failure_examples"):
            lines.extend([
                "## Failure Examples",
                "",
            ])
            for category, examples in data["failure_examples"].items():
                lines.append(f"### {category.replace('_', ' ').title()}")
                lines.append("")
                for ex in examples:
                    lines.append(f"**Query:** {ex['query']}")
                    lines.append(f"- Provider: {ex['provider']}")
                    lines.append(f"- Score: {ex['score']:.2f}")
                    lines.append(f"- Failures: {', '.join(ex['failure_modes'])}")
                    lines.append("")

        # Footer
        lines.extend([
            "---",
            "*Generated by SearchProbe*",
        ])

        # Write file
        output_path = self.output_dir / f"report_{run_id[:8]}.md"
        output_path.write_text("\n".join(lines))

        return output_path

    def _generate_html(self, data: dict[str, Any]) -> Path:
        """Generate HTML report with embedded charts.

        Args:
            data: Report data

        Returns:
            Path to output file
        """
        run_id = data["run_id"]
        stats = data["stats"]
        chart_data = data["chart_data"]

        # Generate charts
        charts_html = {}

        # Radar chart
        if chart_data:
            radar = create_radar_chart(chart_data)
            charts_html["radar"] = radar.to_html(full_html=False, include_plotlyjs="cdn")

        # Heatmap
        if chart_data:
            heatmap = create_heatmap(chart_data)
            charts_html["heatmap"] = heatmap.to_html(full_html=False, include_plotlyjs=False)

        # Bar chart of overall scores
        if stats.get("by_provider"):
            bar_data = {k: v.mean for k, v in stats["by_provider"].items()}
            error_bars = {k: (v.lower, v.upper) for k, v in stats["by_provider"].items()}
            bar = create_bar_chart(bar_data, error_bars=error_bars)
            charts_html["bar"] = bar.to_html(full_html=False, include_plotlyjs=False)

        # Failure mode chart
        if stats.get("failure_modes"):
            failure_chart = create_failure_mode_chart(stats["failure_modes"])
            charts_html["failures"] = failure_chart.to_html(full_html=False, include_plotlyjs=False)

        # Cost breakdown
        if data.get("cost_data"):
            cost_chart = create_cost_breakdown(data["cost_data"])
            charts_html["cost"] = cost_chart.to_html(full_html=False, include_plotlyjs=False)

        # Build HTML
        html = self._build_html_report(data, charts_html)

        # Write file
        output_path = self.output_dir / f"report_{run_id[:8]}.html"
        output_path.write_text(html)

        return output_path

    def _build_html_report(
        self,
        data: dict[str, Any],
        charts: dict[str, str],
    ) -> str:
        """Build HTML report content.

        Args:
            data: Report data
            charts: Dict of chart_name -> HTML

        Returns:
            Complete HTML string
        """
        run_id = data["run_id"]
        stats = data["stats"]
        run_stats = data.get("run_stats", {})

        # Build HTML manually (template fallback)
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SearchProbe Report - {run_id[:8]}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
            color: #333;
        }}
        h1, h2, h3 {{ color: #1a1a2e; }}
        .card {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
        }}
        .stat-box {{
            text-align: center;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 6px;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #636EFA;
        }}
        .stat-label {{
            color: #666;
            font-size: 0.9em;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{
            background: #f8f9fa;
            font-weight: 600;
        }}
        .chart-container {{
            margin: 20px 0;
        }}
        .failure-example {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 10px 15px;
            margin: 10px 0;
        }}
        footer {{
            text-align: center;
            color: #666;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
        }}
    </style>
</head>
<body>
    <h1>SearchProbe Benchmark Report</h1>
    <p><strong>Run ID:</strong> <code>{run_id}</code></p>
    <p><strong>Generated:</strong> {data['generated_at']}</p>

    <div class="card">
        <h2>Overall Statistics</h2>
        <div class="stats-grid">
            <div class="stat-box">
                <div class="stat-value">{stats.get('count', 0)}</div>
                <div class="stat-label">Total Evaluated</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{stats.get('mean', 0):.3f}</div>
                <div class="stat-label">Mean Score</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{stats.get('median', 0):.3f}</div>
                <div class="stat-label">Median Score</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">${run_stats.get('cost_total', 0):.4f}</div>
                <div class="stat-label">Total Cost</div>
            </div>
        </div>
    </div>
"""

        # Add radar chart
        if charts.get("radar"):
            html += f"""
    <div class="card">
        <h2>Provider Performance by Category</h2>
        <div class="chart-container">
            {charts['radar']}
        </div>
    </div>
"""

        # Add heatmap
        if charts.get("heatmap"):
            html += f"""
    <div class="card">
        <h2>Score Heatmap</h2>
        <div class="chart-container">
            {charts['heatmap']}
        </div>
    </div>
"""

        # Add bar chart
        if charts.get("bar"):
            html += f"""
    <div class="card">
        <h2>Overall Scores by Provider</h2>
        <div class="chart-container">
            {charts['bar']}
        </div>
    </div>
"""

        # Add failure mode chart
        if charts.get("failures"):
            html += f"""
    <div class="card">
        <h2>Top Failure Modes</h2>
        <div class="chart-container">
            {charts['failures']}
        </div>
    </div>
"""

        # Add provider table
        if stats.get("by_provider"):
            html += """
    <div class="card">
        <h2>Scores by Provider</h2>
        <table>
            <thead>
                <tr>
                    <th>Provider</th>
                    <th>Mean Score</th>
                    <th>95% CI</th>
                    <th>n</th>
                </tr>
            </thead>
            <tbody>
"""
            for provider, ci in stats["by_provider"].items():
                html += f"""
                <tr>
                    <td>{provider}</td>
                    <td>{ci.mean:.3f}</td>
                    <td>[{ci.lower:.3f}, {ci.upper:.3f}]</td>
                    <td>{ci.n}</td>
                </tr>
"""
            html += """
            </tbody>
        </table>
    </div>
"""

        # Add failure examples
        if data.get("failure_examples"):
            html += """
    <div class="card">
        <h2>Failure Examples</h2>
"""
            for category, examples in data["failure_examples"].items():
                html += f"<h3>{category.replace('_', ' ').title()}</h3>"
                for ex in examples:
                    modes = ", ".join(ex["failure_modes"])
                    html += f"""
        <div class="failure-example">
            <strong>Query:</strong> {ex['query']}<br>
            <strong>Provider:</strong> {ex['provider']} |
            <strong>Score:</strong> {ex['score']:.2f} |
            <strong>Failures:</strong> {modes}
        </div>
"""
            html += "</div>"

        # Footer
        html += """
    <footer>
        <p>Generated by SearchProbe - Adversarial Benchmark for Neural Search</p>
    </footer>
</body>
</html>
"""

        return html
