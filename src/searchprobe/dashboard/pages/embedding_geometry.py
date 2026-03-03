"""Dashboard page for embedding geometry analysis."""

import streamlit as st

from searchprobe.reporting.charts import (
    create_similarity_distribution,
    create_vulnerability_heatmap,
)
from searchprobe.storage import Database


def render(db: Database, run_id: str) -> None:
    """Render the embedding geometry analysis page."""
    st.header("Embedding Geometry Analysis")
    st.markdown(
        "Analyze the geometric properties of embedding spaces to understand "
        "**why** search fails on adversarial queries."
    )

    # Get geometry results from database
    geometry_results = db.get_geometry_results()

    if not geometry_results:
        st.info(
            "No geometry analysis results found. "
            "Run `searchprobe geometry` to analyze embedding spaces."
        )
        _show_explanation()
        return

    # Group by model
    models: dict[str, list[dict]] = {}
    for result in geometry_results:
        model = result["model_name"]
        if model not in models:
            models[model] = []
        models[model].append(result)

    # Vulnerability heatmap
    st.subheader("Vulnerability Heatmap")
    vulnerability_data: dict[str, dict[str, float]] = {}
    for model, results in models.items():
        vulnerability_data[model] = {}
        for r in results:
            vulnerability_data[model][r["category"]] = r.get("vulnerability_score", 0)

    fig = create_vulnerability_heatmap(vulnerability_data)
    st.plotly_chart(fig, use_container_width=True)

    # Model selector
    selected_model = st.selectbox("Select Model", list(models.keys()))

    if selected_model:
        model_results = models[selected_model]

        # Category details
        st.subheader(f"Category Details — {selected_model}")

        cols = st.columns(3)
        for i, result in enumerate(sorted(model_results, key=lambda x: x.get("vulnerability_score", 0), reverse=True)):
            col = cols[i % 3]
            with col:
                vuln = result.get("vulnerability_score", 0)
                color = "red" if vuln > 0.6 else "orange" if vuln > 0.3 else "green"
                st.metric(
                    label=result["category"].replace("_", " ").title(),
                    value=f"{vuln:.2f}",
                    delta=f"Collapse: {result.get('collapse_ratio', 0):.2f}x",
                )

        # Similarity distribution
        st.subheader("Similarity Distributions")
        adv_sims = []
        baseline_sims = []
        random_sims = []

        for result in model_results:
            pair_details = result.get("pair_details", {})
            if isinstance(pair_details, dict):
                adv_sims.extend(pair_details.get("adversarial", []))
                baseline_sims.extend(pair_details.get("baseline", []))
                random_sims.extend(pair_details.get("random", []))

        if adv_sims:
            fig = create_similarity_distribution(adv_sims, baseline_sims, random_sims)
            st.plotly_chart(fig, use_container_width=True)

        # Pair explorer
        st.subheader("Adversarial Pair Explorer")
        category_filter = st.selectbox(
            "Filter by Category",
            ["All"] + [r["category"] for r in model_results],
        )

        for result in model_results:
            if category_filter != "All" and result["category"] != category_filter:
                continue

            pair_details = result.get("pair_details", {})
            details = pair_details.get("details", []) if isinstance(pair_details, dict) else []

            if details:
                with st.expander(f"{result['category'].replace('_', ' ').title()} — {len(details)} pairs"):
                    for pair in details:
                        sim = pair.get("similarity", 0)
                        color = "red" if sim > 0.85 else "orange" if sim > 0.7 else "green"
                        st.markdown(
                            f"**{pair.get('query_a', '')}** vs **{pair.get('query_b', '')}**  \n"
                            f"Similarity: :{color}[{sim:.4f}]"
                        )


def _show_explanation() -> None:
    """Show explanation of embedding geometry analysis."""
    with st.expander("What is Embedding Geometry Analysis?"):
        st.markdown("""
        **Key Insight:** If `cos(embed("companies in AI"), embed("companies NOT in AI")) = 0.96`,
        then **any** embedding-based retrieval system will fail on negation — it's a fundamental
        geometric limitation, not a provider bug.

        ### Metrics
        - **Vulnerability Score** (0-1): Composite metric combining collapse ratio and baseline similarity
        - **Collapse Ratio**: `adversarial_sim / baseline_sim` — values >1 mean adversarial pairs
          are MORE similar than they should be
        - **Intrinsic Dimensionality**: MLE estimate of local dimensionality (lower = more clustered)
        - **Isotropy Score**: How uniformly embeddings are distributed (higher = better)

        ### How to run
        ```bash
        searchprobe geometry --models all-MiniLM-L6-v2,all-mpnet-base-v2
        ```
        """)
