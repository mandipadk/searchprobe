"""Provider comparison page."""

import streamlit as st
import pandas as pd

from searchprobe.evaluation.statistics import (
    aggregate_by_category,
    compare_providers,
)
from searchprobe.reporting.charts import create_heatmap, create_radar_chart
from searchprobe.storage import Database


def render(db: Database, run_id: str) -> None:
    """Render the provider comparison page."""
    st.markdown('<h1 class="main-header">🏆 Provider Comparison</h1>', unsafe_allow_html=True)

    evaluations = db.get_evaluations(run_id)

    if not evaluations:
        st.warning("No evaluations found for this run.")
        return

    # Get unique providers
    providers = list(set(
        f"{e.get('provider', 'unknown')}:{e.get('search_mode', 'default')}"
        for e in evaluations
    ))

    if len(providers) < 2:
        st.info("Need at least 2 providers for comparison.")
        return

    # Provider selection
    col1, col2 = st.columns(2)
    with col1:
        provider_a = st.selectbox("Provider A", providers, index=0)
    with col2:
        other_providers = [p for p in providers if p != provider_a]
        provider_b = st.selectbox("Provider B", other_providers, index=0)

    st.markdown("---")

    # Prepare data
    eval_data = [{
        "category": e.get("category"),
        "provider": e.get("provider"),
        "search_mode": e.get("search_mode"),
        "weighted_score": e.get("weighted_score", 0),
    } for e in evaluations]

    # Head-to-head comparison
    st.subheader("Head-to-Head Comparison")

    scores_a = [
        e["weighted_score"]
        for e in eval_data
        if f"{e['provider']}:{e.get('search_mode', 'default')}" == provider_a
    ]
    scores_b = [
        e["weighted_score"]
        for e in eval_data
        if f"{e['provider']}:{e.get('search_mode', 'default')}" == provider_b
    ]

    if scores_a and scores_b:
        comparison = compare_providers(
            provider_a, scores_a, provider_b, scores_b, paired=False
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            mean_a = sum(scores_a) / len(scores_a)
            st.metric(provider_a, f"{mean_a:.3f}", delta=None)
        with col2:
            mean_b = sum(scores_b) / len(scores_b)
            st.metric(provider_b, f"{mean_b:.3f}", delta=None)
        with col3:
            diff = comparison.mean_diff
            sig = "✓ Significant" if comparison.significant else "Not significant"
            st.metric(
                "Difference (A - B)",
                f"{diff:+.3f}",
                delta=f"p={comparison.p_value:.3f} ({sig})"
            )

    st.markdown("---")

    # Radar comparison
    st.subheader("Performance by Category")

    by_category = aggregate_by_category(eval_data)

    # Build chart data for selected providers
    chart_data = {}
    for category, provider_scores in by_category.items():
        for provider_key, ci in provider_scores.items():
            if provider_key in [provider_a, provider_b]:
                if provider_key not in chart_data:
                    chart_data[provider_key] = {}
                chart_data[provider_key][category] = ci.mean

    if chart_data:
        radar_fig = create_radar_chart(chart_data, title="", height=500)
        st.plotly_chart(radar_fig, use_container_width=True)

    # Heatmap
    st.subheader("Score Heatmap (All Providers)")

    all_chart_data = {}
    for category, provider_scores in by_category.items():
        for provider_key, ci in provider_scores.items():
            if provider_key not in all_chart_data:
                all_chart_data[provider_key] = {}
            all_chart_data[provider_key][category] = ci.mean

    if all_chart_data:
        heatmap_fig = create_heatmap(all_chart_data, title="", height=400)
        st.plotly_chart(heatmap_fig, use_container_width=True)

    # Detailed comparison table
    st.subheader("Category-by-Category Comparison")

    comparison_data = []
    categories = set()
    for category, provider_scores in by_category.items():
        categories.add(category)

    for category in sorted(categories):
        row = {"Category": category}
        if category in by_category:
            for provider_key in [provider_a, provider_b]:
                if provider_key in by_category[category]:
                    ci = by_category[category][provider_key]
                    row[provider_key] = ci.mean
                else:
                    row[provider_key] = None
        comparison_data.append(row)

    if comparison_data:
        df = pd.DataFrame(comparison_data)
        st.dataframe(df.style.format({
            provider_a: "{:.3f}",
            provider_b: "{:.3f}",
        }, na_rep="-"), use_container_width=True)
