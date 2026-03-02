"""Failure analysis page."""

import streamlit as st
import pandas as pd

from searchprobe.evaluation.statistics import failure_mode_frequency
from searchprobe.reporting.charts import create_failure_mode_chart
from searchprobe.storage import Database


def render(db: Database, run_id: str) -> None:
    """Render the failure analysis page."""
    st.markdown('<h1 class="main-header">⚠️ Failure Analysis</h1>', unsafe_allow_html=True)

    evaluations = db.get_evaluations(run_id)

    if not evaluations:
        st.warning("No evaluations found for this run.")
        return

    # Get all failure modes
    all_failures = []
    for e in evaluations:
        for mode in e.get("failure_modes", []):
            all_failures.append({
                "mode": mode,
                "provider": f"{e.get('provider')}:{e.get('search_mode', 'default')}",
                "category": e.get("category"),
                "query": e.get("query_text"),
                "score": e.get("weighted_score", 0),
            })

    if not all_failures:
        st.success("No failures recorded! All results passed evaluation.")
        return

    # Summary metrics
    col1, col2, col3 = st.columns(3)

    unique_modes = set(f["mode"] for f in all_failures)
    with col1:
        st.metric("Total Failures", len(all_failures))
    with col2:
        st.metric("Unique Failure Types", len(unique_modes))
    with col3:
        failure_rate = len(all_failures) / max(len(evaluations), 1) * 100
        st.metric("Failure Rate", f"{failure_rate:.1f}%")

    st.markdown("---")

    # Failure mode distribution
    st.subheader("Failure Mode Distribution")

    failure_counts = failure_mode_frequency([{
        "failure_modes": e.get("failure_modes", [])
    } for e in evaluations])

    if failure_counts:
        fig = create_failure_mode_chart(failure_counts, top_n=15, height=500)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Failure modes by provider
    st.subheader("Failures by Provider")

    provider_failures: dict[str, dict[str, int]] = {}
    for f in all_failures:
        provider = f["provider"]
        mode = f["mode"]
        if provider not in provider_failures:
            provider_failures[provider] = {}
        provider_failures[provider][mode] = provider_failures[provider].get(mode, 0) + 1

    for provider, modes in provider_failures.items():
        with st.expander(f"{provider} ({sum(modes.values())} failures)"):
            df = pd.DataFrame([
                {"Mode": k, "Count": v}
                for k, v in sorted(modes.items(), key=lambda x: x[1], reverse=True)
            ])
            st.dataframe(df, use_container_width=True)

    st.markdown("---")

    # Failure modes by category
    st.subheader("Failures by Category")

    category_failures: dict[str, dict[str, int]] = {}
    for f in all_failures:
        category = f["category"] or "unknown"
        mode = f["mode"]
        if category not in category_failures:
            category_failures[category] = {}
        category_failures[category][mode] = category_failures[category].get(mode, 0) + 1

    for category, modes in sorted(category_failures.items()):
        with st.expander(f"{category.replace('_', ' ').title()} ({sum(modes.values())} failures)"):
            df = pd.DataFrame([
                {"Mode": k, "Count": v}
                for k, v in sorted(modes.items(), key=lambda x: x[1], reverse=True)
            ])
            st.dataframe(df, use_container_width=True)

    st.markdown("---")

    # Worst performing queries
    st.subheader("Queries with Most Failures")

    query_failure_counts: dict[str, int] = {}
    query_details: dict[str, dict] = {}

    for f in all_failures:
        query = f["query"]
        if query:
            query_failure_counts[query] = query_failure_counts.get(query, 0) + 1
            if query not in query_details:
                query_details[query] = {
                    "category": f["category"],
                    "modes": set(),
                }
            query_details[query]["modes"].add(f["mode"])

    # Sort by failure count
    sorted_queries = sorted(query_failure_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    for query, count in sorted_queries:
        details = query_details.get(query, {})
        modes = details.get("modes", set())
        category = details.get("category", "unknown")

        with st.expander(f"{query[:70]}... ({count} failures)"):
            st.write(f"**Full Query:** {query}")
            st.write(f"**Category:** {category.replace('_', ' ').title()}")
            st.write("**Failure Modes:**")
            for m in modes:
                st.write(f"- {m}")
