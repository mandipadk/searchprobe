"""Query explorer page."""

import json
import streamlit as st
import pandas as pd

from searchprobe.storage import Database


def render(db: Database, run_id: str) -> None:
    """Render the query explorer page."""
    st.markdown('<h1 class="main-header">🔎 Query Explorer</h1>', unsafe_allow_html=True)

    evaluations = db.get_evaluations(run_id)

    if not evaluations:
        st.warning("No evaluations found for this run.")
        return

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        categories = sorted(set(e.get("category", "unknown") for e in evaluations))
        selected_category = st.selectbox(
            "Category",
            ["All"] + categories,
        )

    with col2:
        providers = sorted(set(
            f"{e.get('provider', 'unknown')}:{e.get('search_mode', 'default')}"
            for e in evaluations
        ))
        selected_provider = st.selectbox(
            "Provider",
            ["All"] + providers,
        )

    with col3:
        score_filter = st.selectbox(
            "Score Filter",
            ["All", "Low (< 0.5)", "Medium (0.5-0.75)", "High (> 0.75)"],
        )

    # Apply filters
    filtered_evals = evaluations

    if selected_category != "All":
        filtered_evals = [e for e in filtered_evals if e.get("category") == selected_category]

    if selected_provider != "All":
        filtered_evals = [
            e for e in filtered_evals
            if f"{e.get('provider')}:{e.get('search_mode', 'default')}" == selected_provider
        ]

    if score_filter == "Low (< 0.5)":
        filtered_evals = [e for e in filtered_evals if e.get("weighted_score", 0) < 0.5]
    elif score_filter == "Medium (0.5-0.75)":
        filtered_evals = [e for e in filtered_evals if 0.5 <= e.get("weighted_score", 0) <= 0.75]
    elif score_filter == "High (> 0.75)":
        filtered_evals = [e for e in filtered_evals if e.get("weighted_score", 0) > 0.75]

    st.markdown("---")
    st.write(f"Showing {len(filtered_evals)} results")

    # Query list
    for i, e in enumerate(filtered_evals[:50]):  # Limit to 50 for performance
        score = e.get("weighted_score", 0)
        color = "green" if score > 0.75 else "orange" if score > 0.5 else "red"

        with st.expander(
            f"[{e.get('provider')}:{e.get('search_mode', 'default')}] "
            f"{e.get('query_text', '')[:60]}... "
            f"(Score: :{color}[{score:.2f}])"
        ):
            col1, col2 = st.columns([2, 1])

            with col1:
                st.write("**Query:**")
                st.code(e.get("query_text", ""))

                st.write("**Category:**", e.get("category", "unknown").replace("_", " ").title())
                st.write("**Assessment:**", e.get("reasoning", "No assessment available"))

            with col2:
                st.write("**Dimension Scores:**")
                if e.get("scores"):
                    scores = e["scores"]
                    if isinstance(scores, str):
                        scores = json.loads(scores)
                    for dim, score_data in scores.items():
                        if isinstance(score_data, dict):
                            st.write(f"- {dim}: {score_data.get('score', 'N/A')}")
                        else:
                            st.write(f"- {dim}: {score_data}")

                st.write("**Failure Modes:**")
                failures = e.get("failure_modes", [])
                if failures:
                    for f in failures:
                        st.write(f"- ⚠️ {f}")
                else:
                    st.write("✅ No failures")

    if len(filtered_evals) > 50:
        st.info(f"Showing first 50 of {len(filtered_evals)} results. Apply filters to narrow down.")
