"""Category deep dive page."""

import streamlit as st
import pandas as pd

from searchprobe.queries.taxonomy import AdversarialCategory, CATEGORY_METADATA
from searchprobe.evaluation.statistics import aggregate_by_category, calculate_confidence_interval
from searchprobe.reporting.charts import create_bar_chart
from searchprobe.storage import Database


def render(db: Database, run_id: str) -> None:
    """Render the category deep dive page."""
    st.markdown('<h1 class="main-header">📁 Category Deep Dive</h1>', unsafe_allow_html=True)

    evaluations = db.get_evaluations(run_id)

    if not evaluations:
        st.warning("No evaluations found for this run.")
        return

    # Get unique categories
    categories = sorted(set(e.get("category", "unknown") for e in evaluations))

    # Category selection
    selected_category = st.selectbox(
        "Select Category",
        categories,
        format_func=lambda x: x.replace("_", " ").title()
    )

    st.markdown("---")

    # Show category metadata
    try:
        cat_enum = AdversarialCategory(selected_category)
        metadata = CATEGORY_METADATA.get(cat_enum, {})
        if metadata:
            with st.expander("Category Information", expanded=True):
                st.write(f"**Description:** {metadata.get('description', 'N/A')}")
                st.write(f"**Failure Hypothesis:** {metadata.get('failure_hypothesis', 'N/A')}")
                st.write(f"**Difficulty:** {metadata.get('difficulty', 'N/A')}")
    except ValueError:
        pass

    st.markdown("---")

    # Filter evaluations for this category
    category_evals = [e for e in evaluations if e.get("category") == selected_category]

    # Summary metrics
    col1, col2, col3 = st.columns(3)

    scores = [e.get("weighted_score", 0) for e in category_evals]
    ci = calculate_confidence_interval(scores)

    with col1:
        st.metric("Queries", len(category_evals))
    with col2:
        st.metric("Mean Score", f"{ci.mean:.3f}")
    with col3:
        st.metric("95% CI", f"[{ci.lower:.3f}, {ci.upper:.3f}]")

    st.markdown("---")

    # Scores by provider for this category
    st.subheader("Performance by Provider")

    provider_scores: dict[str, list[float]] = {}
    for e in category_evals:
        provider_key = f"{e.get('provider', 'unknown')}:{e.get('search_mode', 'default')}"
        if provider_key not in provider_scores:
            provider_scores[provider_key] = []
        provider_scores[provider_key].append(e.get("weighted_score", 0))

    if provider_scores:
        bar_data = {k: sum(v) / len(v) for k, v in provider_scores.items()}
        bar_fig = create_bar_chart(
            bar_data,
            title=f"Scores for {selected_category.replace('_', ' ').title()}",
            height=400,
        )
        st.plotly_chart(bar_fig, use_container_width=True)

    # Detailed results table
    st.subheader("Individual Results")

    results_data = []
    for e in category_evals:
        results_data.append({
            "Provider": f"{e.get('provider', 'unknown')}:{e.get('search_mode', 'default')}",
            "Query": e.get("query_text", "")[:80] + "...",
            "Score": e.get("weighted_score", 0),
            "Failures": ", ".join(e.get("failure_modes", [])[:3]),
        })

    if results_data:
        df = pd.DataFrame(results_data)
        # Sort by score
        df = df.sort_values("Score", ascending=True)
        st.dataframe(df.style.format({"Score": "{:.3f}"}), use_container_width=True)

    # Failure mode distribution for this category
    st.subheader("Failure Modes in This Category")

    failure_counts: dict[str, int] = {}
    for e in category_evals:
        for mode in e.get("failure_modes", []):
            failure_counts[mode] = failure_counts.get(mode, 0) + 1

    if failure_counts:
        failure_df = pd.DataFrame([
            {"Failure Mode": k, "Count": v}
            for k, v in sorted(failure_counts.items(), key=lambda x: x[1], reverse=True)
        ])
        st.dataframe(failure_df, use_container_width=True)
    else:
        st.info("No failures recorded for this category.")
