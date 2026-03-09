"""Report generator for benchmark results."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
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
    create_vulnerability_heatmap,
    create_embedding_gap_chart,
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
        self.settings = settings or get_settings()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        try:
            self.env = Environment(
                loader=PackageLoader("searchprobe.reporting", "templates"),
                autoescape=select_autoescape(["html", "xml"]),
            )
        except Exception:
            self.env = None

    def generate(
        self,
        run_id: str,
        format: str = "both",
        db: Database | None = None,
    ) -> dict[str, Path]:
        db = db or Database(self.settings.database_path)

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

        chart_data: dict[str, dict[str, float]] = {}
        for category, providers in by_category.items():
            for provider_key, ci in providers.items():
                if provider_key not in chart_data:
                    chart_data[provider_key] = {}
                chart_data[provider_key][category] = ci.mean

        failure_examples = self._get_failure_examples(evaluations)

        cost_data = {}
        if run_stats.get("providers"):
            for p in run_stats["providers"]:
                cost_data[p["name"]] = p.get("cost", 0) or 0

        # --- Geometry data ---
        geometry_data = None
        geo_results = db.get_geometry_results(run_id=run_id)
        if not geo_results:
            # Fall back to all geometry results if none scoped to this run
            geo_results = db.get_geometry_results()
        if geo_results:
            vulnerability_map: dict[str, dict[str, float]] = {}
            all_models = set()
            all_geo_categories = set()
            scores = []
            for r in geo_results:
                model = r.get("model_name", "unknown")
                cat = r.get("category", "unknown")
                score = r.get("vulnerability_score", 0) or 0
                all_models.add(model)
                all_geo_categories.add(cat)
                scores.append(score)
                if model not in vulnerability_map:
                    vulnerability_map[model] = {}
                vulnerability_map[model][cat] = score

            mean_vuln = sum(scores) / len(scores) if scores else 0
            # Find most vulnerable category
            cat_scores: dict[str, list[float]] = {}
            for r in geo_results:
                cat = r["category"]
                if cat not in cat_scores:
                    cat_scores[cat] = []
                cat_scores[cat].append(r.get("vulnerability_score", 0) or 0)
            most_vulnerable_cat = max(cat_scores, key=lambda c: sum(cat_scores[c]) / len(cat_scores[c])) if cat_scores else "N/A"

            geometry_data = {
                "vulnerability_map": vulnerability_map,
                "models": sorted(all_models),
                "categories": sorted(all_geo_categories),
                "mean_vulnerability": mean_vuln,
                "most_vulnerable_category": most_vulnerable_cat,
                "num_models": len(all_models),
                "num_categories": len(all_geo_categories),
            }

        # --- Perturbation data ---
        perturbation_data = None
        perturb_results = db.get_perturbation_results_with_category(run_id)
        if perturb_results:
            operator_scores: dict[str, list[float]] = {}
            category_scores: dict[str, list[float]] = {}
            all_jaccard = []
            all_rbo = []
            for r in perturb_results:
                op = r.get("operator", "unknown")
                cat = r.get("category") or "unknown"
                j = r.get("jaccard_similarity", 0) or 0
                rbo = r.get("rbo_score", 0) or 0
                all_jaccard.append(j)
                all_rbo.append(rbo)
                if op not in operator_scores:
                    operator_scores[op] = []
                operator_scores[op].append(j)
                if cat != "unknown":
                    if cat not in category_scores:
                        category_scores[cat] = []
                    category_scores[cat].append(j)

            operator_means = {op: sum(s) / len(s) for op, s in operator_scores.items()}
            category_means = {cat: sum(s) / len(s) for cat, s in category_scores.items()}
            mean_jaccard = sum(all_jaccard) / len(all_jaccard) if all_jaccard else 0
            mean_rbo = sum(all_rbo) / len(all_rbo) if all_rbo else 0
            least_stable_op = min(operator_means, key=operator_means.get) if operator_means else "N/A"

            perturbation_data = {
                "operator_means": operator_means,
                "category_means": category_means,
                "mean_jaccard": mean_jaccard,
                "mean_rbo": mean_rbo,
                "total_perturbations": len(perturb_results),
                "least_stable_operator": least_stable_op,
            }

        # --- Validation data ---
        validation_data = None
        val_results = db.get_validation_results_with_category(run_id)
        if val_results:
            cat_improvements: dict[str, list[float]] = {}
            provider_data: dict[str, dict[str, list[float]]] = {}
            all_ndcg_imp = []
            all_tau = []
            for r in val_results:
                cat = r.get("category", "unknown")
                prov = r.get("provider", "unknown")
                imp = r.get("ndcg_improvement", 0) or 0
                tau = r.get("kendall_tau", 0) or 0
                all_ndcg_imp.append(imp)
                all_tau.append(tau)
                if cat not in cat_improvements:
                    cat_improvements[cat] = []
                cat_improvements[cat].append(imp)
                if prov not in provider_data:
                    provider_data[prov] = {"original_ndcg": [], "reranked_ndcg": []}
                provider_data[prov]["original_ndcg"].append(r.get("original_ndcg", 0) or 0)
                provider_data[prov]["reranked_ndcg"].append(r.get("reranked_ndcg", 0) or 0)

            category_improvement_means = {cat: sum(v) / len(v) for cat, v in cat_improvements.items()}
            mean_ndcg_imp = sum(all_ndcg_imp) / len(all_ndcg_imp) if all_ndcg_imp else 0
            max_ndcg_imp = max(all_ndcg_imp) if all_ndcg_imp else 0
            mean_tau = sum(all_tau) / len(all_tau) if all_tau else 0

            provider_means = {}
            for prov, d in provider_data.items():
                provider_means[prov] = {
                    "original_ndcg": sum(d["original_ndcg"]) / len(d["original_ndcg"]),
                    "reranked_ndcg": sum(d["reranked_ndcg"]) / len(d["reranked_ndcg"]),
                }

            validation_data = {
                "category_improvements": category_improvement_means,
                "provider_means": provider_means,
                "mean_ndcg_improvement": mean_ndcg_imp,
                "max_ndcg_improvement": max_ndcg_imp,
                "mean_kendall_tau": mean_tau,
                "queries_validated": len(val_results),
            }

        # --- Evolution data ---
        evolution_data = None
        evo_results = db.get_evolution_results()
        if evo_results:
            latest = evo_results[0]
            best_individuals = latest.get("best_individuals", [])
            fitness_history = latest.get("fitness_history", [])
            best_fitness = max((ind.get("fitness", 0) for ind in best_individuals), default=0)

            evolution_data = {
                "generations": latest.get("generations_completed", 0),
                "total_evaluations": latest.get("total_evaluations", 0),
                "best_fitness": best_fitness,
                "fitness_mode": latest.get("fitness_mode", "unknown"),
                "best_individuals": best_individuals[:10],
                "fitness_history": fitness_history,
                "total_cost": latest.get("total_cost", 0),
            }

        return {
            "run_id": run_id,
            "run_stats": run_stats,
            "evaluations": evaluations,
            "stats": stats,
            "chart_data": chart_data,
            "failure_examples": failure_examples,
            "cost_data": cost_data,
            "geometry_data": geometry_data,
            "perturbation_data": perturbation_data,
            "validation_data": validation_data,
            "evolution_data": evolution_data,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def _get_failure_examples(
        self,
        evaluations: list[dict],
        max_per_category: int = 3,
    ) -> dict[str, list[dict]]:
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

        if run_stats:
            lines.extend([
                "## Run Summary",
                "",
                f"- **Started:** {run_stats.get('started_at', 'N/A')}",
                f"- **Completed:** {run_stats.get('completed_at', 'N/A')}",
                f"- **Total Cost:** ${run_stats.get('cost_total', 0):.4f}",
                "",
            ])

        lines.extend([
            "## Overall Statistics",
            "",
            f"- **Total Evaluated:** {stats.get('count', 0)}",
            f"- **Mean Score:** {stats.get('mean', 0):.3f}",
            f"- **Median Score:** {stats.get('median', 0):.3f}",
            f"- **Std Deviation:** {stats.get('std', 0):.3f}",
            "",
        ])

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

        if stats.get("by_category"):
            lines.extend(["## Scores by Category", ""])
            all_providers = set()
            for cat_data in stats["by_category"].values():
                all_providers.update(cat_data.keys())
            providers = sorted(all_providers)

            header = "| Category | " + " | ".join(providers) + " |"
            separator = "|----------|" + " | ".join(["------"] * len(providers)) + " |"
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

        # Geometry section
        if data.get("geometry_data"):
            gd = data["geometry_data"]
            lines.extend([
                "## Embedding Geometry Analysis",
                "",
                f"- **Models Analyzed:** {gd['num_models']}",
                f"- **Mean Vulnerability:** {gd['mean_vulnerability']:.3f}",
                f"- **Most Vulnerable Category:** {gd['most_vulnerable_category']}",
                "",
            ])
            if gd["vulnerability_map"]:
                cats = gd["categories"]
                models = gd["models"]
                header = "| Category | " + " | ".join(models) + " |"
                sep = "|----------|" + " | ".join(["------"] * len(models)) + " |"
                lines.extend([header, sep])
                for cat in cats:
                    row = f"| {cat.replace('_', ' ').title()} |"
                    for model in models:
                        score = gd["vulnerability_map"].get(model, {}).get(cat, 0)
                        row += f" {score:.2f} |"
                    lines.append(row)
                lines.append("")

        # Perturbation section
        if data.get("perturbation_data"):
            pd = data["perturbation_data"]
            lines.extend([
                "## Perturbation Robustness",
                "",
                f"- **Total Perturbations:** {pd['total_perturbations']}",
                f"- **Mean Jaccard:** {pd['mean_jaccard']:.3f}",
                f"- **Mean RBO:** {pd['mean_rbo']:.3f}",
                f"- **Least Stable Operator:** {pd['least_stable_operator']}",
                "",
            ])
            if pd["operator_means"]:
                lines.extend(["### Stability by Operator", "", "| Operator | Mean Jaccard |", "|----------|-------------|"])
                for op, score in sorted(pd["operator_means"].items(), key=lambda x: x[1]):
                    lines.append(f"| {op.replace('_', ' ').title()} | {score:.3f} |")
                lines.append("")
            if pd["category_means"]:
                lines.extend(["### Stability by Category", "", "| Category | Mean Jaccard |", "|----------|-------------|"])
                for cat, score in sorted(pd["category_means"].items(), key=lambda x: x[1]):
                    lines.append(f"| {cat.replace('_', ' ').title()} | {score:.3f} |")
                lines.append("")

        # Validation section
        if data.get("validation_data"):
            vd = data["validation_data"]
            lines.extend([
                "## Cross-Encoder Validation",
                "",
                f"- **Mean NDCG Improvement:** {vd['mean_ndcg_improvement']:.3f}",
                f"- **Max NDCG Improvement:** {vd['max_ndcg_improvement']:.3f}",
                f"- **Mean Kendall's Tau:** {vd['mean_kendall_tau']:.3f}",
                f"- **Queries Validated:** {vd['queries_validated']}",
                "",
            ])
            if vd["category_improvements"]:
                lines.extend(["### NDCG Improvement by Category", "", "| Category | Mean Improvement |", "|----------|-----------------|"])
                for cat, imp in sorted(vd["category_improvements"].items(), key=lambda x: x[1], reverse=True):
                    lines.append(f"| {cat.replace('_', ' ').title()} | {imp:.3f} |")
                lines.append("")
            if vd["provider_means"]:
                lines.extend(["### Provider Comparison", "", "| Provider | Original NDCG | Reranked NDCG |", "|----------|--------------|---------------|"])
                for prov, means in vd["provider_means"].items():
                    lines.append(f"| {prov} | {means['original_ndcg']:.3f} | {means['reranked_ndcg']:.3f} |")
                lines.append("")

        # Evolution section
        if data.get("evolution_data"):
            ed = data["evolution_data"]
            lines.extend([
                "## Adversarial Evolution",
                "",
                f"- **Generations:** {ed['generations']}",
                f"- **Total Evaluations:** {ed['total_evaluations']}",
                f"- **Best Fitness:** {ed['best_fitness']:.3f}",
                f"- **Fitness Mode:** {ed['fitness_mode']}",
                "",
            ])
            if ed["best_individuals"]:
                lines.extend(["### Top Evolved Queries", "", "| # | Query | Fitness | Category | Mutations |", "|---|-------|---------|----------|-----------|"])
                for i, ind in enumerate(ed["best_individuals"][:10]):
                    query = ind.get("query", "")[:50]
                    fitness = ind.get("fitness", 0)
                    cat = ind.get("category", "-")
                    mutations = len(ind.get("mutation_history", []))
                    lines.append(f"| {i + 1} | {query} | {fitness:.3f} | {cat} | {mutations} |")
                lines.append("")

        # Failure modes
        if stats.get("failure_modes"):
            lines.extend(["## Top Failure Modes", ""])
            for mode, count in list(stats["failure_modes"].items())[:10]:
                lines.append(f"- **{mode}:** {count}")
            lines.append("")

        if data.get("failure_examples"):
            lines.extend(["## Failure Examples", ""])
            for category, examples in data["failure_examples"].items():
                lines.append(f"### {category.replace('_', ' ').title()}")
                lines.append("")
                for ex in examples:
                    lines.append(f"**Query:** {ex['query']}")
                    lines.append(f"- Provider: {ex['provider']}")
                    lines.append(f"- Score: {ex['score']:.2f}")
                    lines.append(f"- Failures: {', '.join(ex['failure_modes'])}")
                    lines.append("")

        lines.extend(["---", "*Generated by SearchProbe*"])

        output_path = self.output_dir / f"report_{run_id[:8]}.md"
        output_path.write_text("\n".join(lines))

        return output_path

    def _generate_html(self, data: dict[str, Any]) -> Path:
        run_id = data["run_id"]
        stats = data["stats"]
        chart_data = data["chart_data"]

        charts_html = {}

        # Radar chart (first one includes plotly.js)
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

        # --- Geometry chart ---
        if data.get("geometry_data"):
            vuln_map = data["geometry_data"]["vulnerability_map"]
            vuln_heatmap = create_vulnerability_heatmap(vuln_map)
            charts_html["geometry_heatmap"] = vuln_heatmap.to_html(full_html=False, include_plotlyjs=False)

        # --- Perturbation charts ---
        if data.get("perturbation_data"):
            pd = data["perturbation_data"]
            # Operator stability bar chart
            if pd["operator_means"]:
                sorted_ops = sorted(pd["operator_means"].items(), key=lambda x: x[1])
                op_labels = [op.replace("_", " ").title() for op, _ in sorted_ops]
                op_values = [v for _, v in sorted_ops]
                op_colors = [
                    "#ef4444" if v < 0.3 else "#f59e0b" if v < 0.6 else "#22c55e"
                    for v in op_values
                ]
                fig = go.Figure(data=[go.Bar(
                    x=op_labels, y=op_values, marker_color=op_colors,
                    text=[f"{v:.2f}" for v in op_values], textposition="outside",
                )])
                fig.update_layout(
                    title="Stability by Perturbation Operator",
                    yaxis=dict(range=[0, 1], title="Mean Jaccard Similarity"),
                    height=400, width=700,
                )
                charts_html["perturbation_operator"] = fig.to_html(full_html=False, include_plotlyjs=False)

            # Category stability bar chart
            if pd["category_means"]:
                sorted_cats = sorted(pd["category_means"].items(), key=lambda x: x[1])
                cat_labels = [c.replace("_", " ").title() for c, _ in sorted_cats]
                cat_values = [v for _, v in sorted_cats]
                fig = go.Figure(data=[go.Bar(
                    x=cat_labels, y=cat_values, marker_color="#6366f1",
                    text=[f"{v:.2f}" for v in cat_values], textposition="outside",
                )])
                fig.update_layout(
                    title="Stability by Category",
                    yaxis=dict(range=[0, 1], title="Mean Jaccard Similarity"),
                    xaxis=dict(tickangle=-45), height=400, width=700,
                )
                charts_html["perturbation_category"] = fig.to_html(full_html=False, include_plotlyjs=False)

        # --- Validation charts ---
        if data.get("validation_data"):
            vd = data["validation_data"]
            # NDCG improvement by category
            if vd["category_improvements"]:
                gap_chart = create_embedding_gap_chart(vd["category_improvements"])
                charts_html["validation_ndcg"] = gap_chart.to_html(full_html=False, include_plotlyjs=False)

            # Provider comparison grouped bar
            if vd["provider_means"]:
                provs = list(vd["provider_means"].keys())
                orig_vals = [vd["provider_means"][p]["original_ndcg"] for p in provs]
                reranked_vals = [vd["provider_means"][p]["reranked_ndcg"] for p in provs]
                fig = go.Figure(data=[
                    go.Bar(name="Original NDCG", x=provs, y=orig_vals, marker_color="#94a3b8"),
                    go.Bar(name="Reranked NDCG", x=provs, y=reranked_vals, marker_color="#6366f1"),
                ])
                fig.update_layout(
                    barmode="group", title="Provider NDCG: Original vs Reranked",
                    yaxis=dict(range=[0, 1], title="NDCG Score"),
                    height=400, width=600,
                )
                charts_html["validation_provider"] = fig.to_html(full_html=False, include_plotlyjs=False)

        # --- Evolution chart ---
        if data.get("evolution_data"):
            ed = data["evolution_data"]
            if ed["fitness_history"]:
                gens = list(range(len(ed["fitness_history"])))
                means = [h.get("mean", 0) for h in ed["fitness_history"]]
                maxes = [h.get("max", 0) for h in ed["fitness_history"]]
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=gens, y=means, mode="lines+markers", name="Mean Fitness",
                    line=dict(color="#6366f1"),
                ))
                fig.add_trace(go.Scatter(
                    x=gens, y=maxes, mode="lines+markers", name="Max Fitness",
                    line=dict(color="#ef4444"),
                ))
                fig.update_layout(
                    title="Fitness History Across Generations",
                    xaxis_title="Generation", yaxis_title="Fitness",
                    height=400, width=700,
                )
                charts_html["evolution_fitness"] = fig.to_html(full_html=False, include_plotlyjs=False)

        html = self._build_html_report(data, charts_html)

        output_path = self.output_dir / f"report_{run_id[:8]}.html"
        output_path.write_text(html)

        return output_path

    def _build_html_report(
        self,
        data: dict[str, Any],
        charts: dict[str, str],
    ) -> str:
        run_id = data["run_id"]
        stats = data["stats"]
        run_stats = data.get("run_stats", {})

        def _stat_box(value: str, label: str) -> str:
            return f"""<div class="bg-stone-50 rounded-lg p-4 text-center">
                <div class="text-stone-800 text-3xl font-bold">{value}</div>
                <div class="text-stone-500 text-sm mt-1">{label}</div>
            </div>"""

        def _section(title: str, subtitle: str, content: str) -> str:
            return f"""<div class="bg-white rounded-xl shadow-sm border border-stone-200 p-6">
                <h2 class="text-xl font-semibold text-stone-900">{title}</h2>
                <p class="text-stone-500 text-sm mt-1 mb-4">{subtitle}</p>
                {content}
            </div>"""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SearchProbe Report - {run_id[:8]}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <script>
        tailwind.config = {{
            theme: {{
                extend: {{
                    fontFamily: {{
                        sans: ['Inter', 'system-ui', 'sans-serif'],
                    }},
                }},
            }},
        }}
    </script>
</head>
<body class="bg-stone-50 font-sans text-stone-700 antialiased">
    <div class="max-w-6xl mx-auto px-4 py-8 space-y-6">
"""

        # Header card
        html += f"""
        <div class="bg-white rounded-xl shadow-sm border border-stone-200 p-6">
            <h1 class="text-2xl font-bold text-stone-900">SearchProbe Benchmark Report</h1>
            <div class="flex flex-wrap gap-x-6 gap-y-1 mt-2 text-sm text-stone-500">
                <span>Run ID: <code class="bg-stone-100 px-1.5 py-0.5 rounded text-stone-700">{run_id[:8]}</code></span>
                <span>Generated: {data['generated_at']}</span>
            </div>
        </div>
"""

        # Overall Statistics
        stats_grid = f"""<div class="grid grid-cols-2 md:grid-cols-4 gap-4">
            {_stat_box(str(stats.get('count', 0)), 'Total Evaluated')}
            {_stat_box(f"{stats.get('mean', 0):.3f}", 'Mean Score')}
            {_stat_box(f"{stats.get('median', 0):.3f}", 'Median Score')}
            {_stat_box(f"${run_stats.get('cost_total', 0):.4f}", 'Total Cost')}
        </div>"""
        html += _section("Overall Statistics", "Summary metrics across all providers and categories", stats_grid)

        # Radar chart
        if charts.get("radar"):
            html += _section(
                "Provider Performance by Category",
                "Radar comparison of provider scores across adversarial categories",
                f'<div>{charts["radar"]}</div>',
            )

        # Heatmap
        if charts.get("heatmap"):
            html += _section(
                "Score Heatmap",
                "Provider scores by category with color intensity",
                f'<div>{charts["heatmap"]}</div>',
            )

        # Bar chart
        if charts.get("bar"):
            html += _section(
                "Overall Scores by Provider",
                "Mean scores with 95% confidence intervals",
                f'<div>{charts["bar"]}</div>',
            )

        # --- Geometry section ---
        if data.get("geometry_data"):
            gd = data["geometry_data"]
            geo_stats = f"""<div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                {_stat_box(str(gd['num_models']), 'Models Analyzed')}
                {_stat_box(f"{gd['mean_vulnerability']:.3f}", 'Mean Vulnerability')}
                {_stat_box(gd['most_vulnerable_category'].replace('_', ' ').title(), 'Most Vulnerable Category')}
                {_stat_box(str(gd['num_categories']), 'Categories Analyzed')}
            </div>"""
            chart_html = charts.get("geometry_heatmap", "")
            html += _section(
                "Embedding Geometry Analysis",
                "Vulnerability scores measuring how adversarial queries collapse embedding space",
                geo_stats + f'<div>{chart_html}</div>',
            )

        # --- Perturbation section ---
        if data.get("perturbation_data"):
            pd = data["perturbation_data"]
            perturb_stats = f"""<div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                {_stat_box(str(pd['total_perturbations']), 'Total Perturbations')}
                {_stat_box(f"{pd['mean_jaccard']:.3f}", 'Mean Jaccard')}
                {_stat_box(f"{pd['mean_rbo']:.3f}", 'Mean RBO')}
                {_stat_box(pd['least_stable_operator'].replace('_', ' ').title(), 'Least Stable Operator')}
            </div>"""
            perturb_charts = ""
            if charts.get("perturbation_operator"):
                perturb_charts += f'<div>{charts["perturbation_operator"]}</div>'
            if charts.get("perturbation_category"):
                perturb_charts += f'<div class="mt-4">{charts["perturbation_category"]}</div>'
            html += _section(
                "Perturbation Robustness",
                "Result stability when queries are systematically perturbed",
                perturb_stats + perturb_charts,
            )

        # --- Validation section ---
        if data.get("validation_data"):
            vd = data["validation_data"]
            val_stats = f"""<div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                {_stat_box(f"{vd['mean_ndcg_improvement']:.3f}", 'Mean NDCG Improvement')}
                {_stat_box(f"{vd['max_ndcg_improvement']:.3f}", 'Max NDCG Improvement')}
                {_stat_box(f"{vd['mean_kendall_tau']:.3f}", "Mean Kendall's Tau")}
                {_stat_box(str(vd['queries_validated']), 'Queries Validated')}
            </div>"""
            val_charts = ""
            if charts.get("validation_ndcg"):
                val_charts += f'<div>{charts["validation_ndcg"]}</div>'
            if charts.get("validation_provider"):
                val_charts += f'<div class="mt-4">{charts["validation_provider"]}</div>'
            html += _section(
                "Cross-Encoder Validation",
                "NDCG improvement when results are reranked by a cross-encoder model",
                val_stats + val_charts,
            )

        # --- Evolution section ---
        if data.get("evolution_data"):
            ed = data["evolution_data"]
            evo_stats = f"""<div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                {_stat_box(str(ed['generations']), 'Generations')}
                {_stat_box(str(ed['total_evaluations']), 'Total Evaluations')}
                {_stat_box(f"{ed['best_fitness']:.3f}", 'Best Fitness')}
                {_stat_box(ed['fitness_mode'].replace('_', ' ').title(), 'Fitness Mode')}
            </div>"""
            evo_charts = ""
            if charts.get("evolution_fitness"):
                evo_charts += f'<div>{charts["evolution_fitness"]}</div>'

            # Top queries table
            if ed["best_individuals"]:
                evo_charts += """<div class="mt-4 overflow-x-auto">
                    <table class="w-full text-sm">
                        <thead>
                            <tr class="border-b border-stone-200">
                                <th class="text-left py-2 px-3 text-stone-600 font-medium">#</th>
                                <th class="text-left py-2 px-3 text-stone-600 font-medium">Query</th>
                                <th class="text-right py-2 px-3 text-stone-600 font-medium">Fitness</th>
                                <th class="text-left py-2 px-3 text-stone-600 font-medium">Category</th>
                                <th class="text-right py-2 px-3 text-stone-600 font-medium">Mutations</th>
                            </tr>
                        </thead>
                        <tbody>"""
                for i, ind in enumerate(ed["best_individuals"][:10]):
                    query = _html_escape(ind.get("query", "")[:60])
                    fitness = ind.get("fitness", 0)
                    cat = ind.get("category", "-")
                    mutations = len(ind.get("mutation_history", []))
                    evo_charts += f"""
                            <tr class="border-b border-stone-100 hover:bg-stone-50">
                                <td class="py-2 px-3 text-stone-400">{i + 1}</td>
                                <td class="py-2 px-3 text-stone-800 font-mono text-xs">{query}</td>
                                <td class="py-2 px-3 text-right font-medium">{fitness:.3f}</td>
                                <td class="py-2 px-3 text-stone-500">{cat}</td>
                                <td class="py-2 px-3 text-right">{mutations}</td>
                            </tr>"""
                evo_charts += """
                        </tbody>
                    </table>
                </div>"""

            html += _section(
                "Adversarial Evolution",
                "Evolutionary optimization of adversarial queries to discover worst-case failures",
                evo_stats + evo_charts,
            )

        # Failure mode chart
        if charts.get("failures"):
            html += _section(
                "Top Failure Modes",
                "Most common failure patterns across all evaluations",
                f'<div>{charts["failures"]}</div>',
            )

        # Cost breakdown
        if charts.get("cost"):
            html += _section(
                "Cost Breakdown",
                "API cost distribution by provider",
                f'<div>{charts["cost"]}</div>',
            )

        # Provider scores table
        if stats.get("by_provider"):
            table_html = """<div class="overflow-x-auto">
                <table class="w-full text-sm">
                    <thead>
                        <tr class="border-b border-stone-200">
                            <th class="text-left py-2 px-3 text-stone-600 font-medium">Provider</th>
                            <th class="text-right py-2 px-3 text-stone-600 font-medium">Mean Score</th>
                            <th class="text-right py-2 px-3 text-stone-600 font-medium">95% CI</th>
                            <th class="text-right py-2 px-3 text-stone-600 font-medium">n</th>
                        </tr>
                    </thead>
                    <tbody>"""
            for provider, ci in stats["by_provider"].items():
                table_html += f"""
                        <tr class="border-b border-stone-100 hover:bg-stone-50">
                            <td class="py-2 px-3 font-medium text-stone-800">{provider}</td>
                            <td class="py-2 px-3 text-right">{ci.mean:.3f}</td>
                            <td class="py-2 px-3 text-right text-stone-500">[{ci.lower:.3f}, {ci.upper:.3f}]</td>
                            <td class="py-2 px-3 text-right text-stone-400">{ci.n}</td>
                        </tr>"""
            table_html += """
                    </tbody>
                </table>
            </div>"""
            html += _section(
                "Scores by Provider",
                "Detailed provider performance with confidence intervals",
                table_html,
            )

        # Failure examples
        if data.get("failure_examples"):
            examples_html = ""
            for category, examples in data["failure_examples"].items():
                examples_html += f'<h3 class="text-lg font-medium text-stone-800 mt-4 mb-2">{category.replace("_", " ").title()}</h3>'
                for ex in examples:
                    modes = ", ".join(ex["failure_modes"])
                    query = _html_escape(ex["query"])
                    examples_html += f"""
                    <div class="bg-amber-50 border-l-4 border-amber-400 rounded-r-lg p-3 mb-2">
                        <div class="font-medium text-stone-800 text-sm">{query}</div>
                        <div class="text-xs text-stone-500 mt-1">
                            {ex['provider']} &middot; Score: {ex['score']:.2f} &middot; {modes}
                        </div>
                    </div>"""
            html += _section(
                "Failure Examples",
                "Representative failure cases by adversarial category",
                examples_html,
            )

        # Footer
        html += """
        <div class="text-center text-stone-400 text-sm py-6 border-t border-stone-200">
            Generated by SearchProbe &mdash; Adversarial Benchmark for Neural Search
        </div>
    </div>
</body>
</html>
"""

        return html


def _html_escape(text: str) -> str:
    """Escape HTML special characters."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
