"""Overview page for the dashboard."""

import streamlit as st
import pandas as pd

from searchprobe.evaluation.statistics import (
    summary_statistics,
    aggregate_by_category,
    aggregate_by_provider,
)
from searchprobe.reporting.charts import create_radar_chart, create_bar_chart
from searchprobe.storage import Database


def render(db: Database, run_id: str) -> None:
    """Render the overview page."""
    st.markdown('<h1 class="main-header">📊 Benchmark Overview</h1>', unsafe_allow_html=True)

    # Get evaluations
    evaluations = db.get_evaluations(run_id)

    if not evaluations:
        st.warning("No evaluations found for this run. Run `searchprobe evaluate` first.")
        _show_run_stats(db, run_id)
        return

    # Calculate statistics
    eval_data = [{
        "category": e.get("category"),
        "provider": e.get("provider"),
        "search_mode": e.get("search_mode"),
        "weighted_score": e.get("weighted_score", 0),
        "failure_modes": e.get("failure_modes", []),
    } for e in evaluations]

    stats = summary_statistics(eval_data)

    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Evaluated", stats.get("count", 0))
    with col2:
        st.metric("Mean Score", f"{stats.get('mean', 0):.3f}")
    with col3:
        st.metric("Median Score", f"{stats.get('median', 0):.3f}")
    with col4:
        run_stats = db.get_run_stats(run_id)
        st.metric("Total Cost", f"${run_stats.get('cost_total', 0):.4f}")

    st.markdown("---")

    # Radar chart
    st.subheader("Provider Performance by Category")

    by_category = aggregate_by_category(eval_data)
    chart_data = {}
    for category, providers in by_category.items():
        for provider_key, ci in providers.items():
            if provider_key not in chart_data:
                chart_data[provider_key] = {}
            chart_data[provider_key][category] = ci.mean

    if chart_data:
        radar_fig = create_radar_chart(chart_data, height=500, width=700)
        st.plotly_chart(radar_fig, use_container_width=True)
    else:
        st.info("Not enough data for radar chart.")

    st.markdown("---")

    # Two-column layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Scores by Provider")
        if stats.get("by_provider"):
            provider_data = []
            for provider, ci in stats["by_provider"].items():
                provider_data.append({
                    "Provider": provider,
                    "Mean": ci.mean,
                    "95% CI Lower": ci.lower,
                    "95% CI Upper": ci.upper,
                    "n": ci.n,
                })
            df = pd.DataFrame(provider_data)
            st.dataframe(df.style.format({
                "Mean": "{:.3f}",
                "95% CI Lower": "{:.3f}",
                "95% CI Upper": "{:.3f}",
            }), use_container_width=True)

            # Bar chart
            bar_data = {k: v.mean for k, v in stats["by_provider"].items()}
            bar_fig = create_bar_chart(bar_data, title="", height=300)
            st.plotly_chart(bar_fig, use_container_width=True)

    with col2:
        st.subheader("Top Failure Modes")
        if stats.get("failure_modes"):
            failure_data = []
            for mode, count in list(stats["failure_modes"].items())[:10]:
                failure_data.append({
                    "Failure Mode": mode,
                    "Count": count,
                })
            df = pd.DataFrame(failure_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No failures recorded.")


def _show_run_stats(db: Database, run_id: str) -> None:
    """Show basic run statistics when no evaluations exist."""
    run_stats = db.get_run_stats(run_id)

    if not run_stats:
        return

    st.subheader("Run Information")

    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Started:** {run_stats.get('started_at', 'N/A')}")
        st.write(f"**Completed:** {run_stats.get('completed_at', 'N/A')}")
    with col2:
        st.write(f"**Total Cost:** ${run_stats.get('cost_total', 0):.4f}")

    if run_stats.get("providers"):
        st.subheader("Results by Provider")
        provider_df = pd.DataFrame(run_stats["providers"])
        st.dataframe(provider_df, use_container_width=True)
