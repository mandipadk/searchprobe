"""Dashboard page for cross-encoder validation and embedding gap analysis."""

import streamlit as st

from searchprobe.reporting.charts import create_embedding_gap_chart
from searchprobe.storage import Database


def render(db: Database, run_id: str) -> None:
    """Render the cross-encoder validation page."""
    st.header("Cross-Encoder Validation")
    st.markdown(
        "Quantify the **embedding gap** — how much relevance is lost by using "
        "bi-encoder search instead of cross-encoder scoring."
    )

    # Get validation results
    validation_results = db.get_validation_results(run_id)

    if not validation_results:
        st.info(
            "No validation results found for this run. "
            "Run `searchprobe validate --run-id <id>` to analyze the embedding gap."
        )
        _show_explanation()
        return

    import numpy as np

    # Summary metrics
    st.subheader("Embedding Gap Summary")
    improvements = [r.get("ndcg_improvement", 0) for r in validation_results]
    taus = [r.get("kendall_tau", 0) for r in validation_results]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean NDCG Improvement", f"{np.mean(improvements):.1%}")
    with col2:
        st.metric("Max NDCG Improvement", f"{np.max(improvements):.1%}")
    with col3:
        st.metric("Mean Kendall's Tau", f"{np.mean(taus):.3f}")
    with col4:
        st.metric("Queries Validated", len(validation_results))

    # NDCG improvement by category
    st.subheader("Embedding Gap by Category")
    category_improvements: dict[str, list[float]] = {}
    for r in validation_results:
        cat = r.get("category", "unknown")
        if cat not in category_improvements:
            category_improvements[cat] = []
        category_improvements[cat].append(r.get("ndcg_improvement", 0))

    mean_by_category = {
        cat: float(np.mean(imps)) for cat, imps in category_improvements.items()
    }

    fig = create_embedding_gap_chart(mean_by_category)
    st.plotly_chart(fig, use_container_width=True)

    # Provider comparison
    st.subheader("Provider Comparison")
    provider_data: dict[str, dict[str, list[float]]] = {}
    for r in validation_results:
        prov = r.get("provider", "unknown")
        if prov not in provider_data:
            provider_data[prov] = {"improvements": [], "taus": []}
        provider_data[prov]["improvements"].append(r.get("ndcg_improvement", 0))
        provider_data[prov]["taus"].append(r.get("kendall_tau", 0))

    if provider_data:
        import plotly.graph_objects as go

        providers = list(provider_data.keys())
        mean_imps = [float(np.mean(d["improvements"])) for d in provider_data.values()]
        mean_taus = [float(np.mean(d["taus"])) for d in provider_data.values()]

        fig = go.Figure(
            data=[
                go.Bar(name="NDCG Improvement", x=providers, y=mean_imps, marker_color="#e74c3c"),
                go.Bar(name="Kendall's Tau", x=providers, y=mean_taus, marker_color="#3498db"),
            ]
        )
        fig.update_layout(
            title="Embedding Gap by Provider",
            barmode="group",
            yaxis_title="Score",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Detailed results
    st.subheader("Detailed Validation Results")
    for r in validation_results[:20]:
        imp = r.get("ndcg_improvement", 0)
        color = "red" if imp > 0.2 else "orange" if imp > 0.1 else "green"
        with st.expander(
            f"[{r.get('provider', '?')}] {r.get('category', '?')} — "
            f"NDCG gap: {imp:.1%}"
        ):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Original NDCG", f"{r.get('original_ndcg', 0):.3f}")
            with col2:
                st.metric("Reranked NDCG", f"{r.get('reranked_ndcg', 0):.3f}")
            with col3:
                st.metric("Kendall's Tau", f"{r.get('kendall_tau', 0):.3f}")


def _show_explanation() -> None:
    """Show explanation of cross-encoder validation."""
    with st.expander("What is Cross-Encoder Validation?"):
        st.markdown("""
        **Cross-encoders** jointly encode (query, document) pairs, producing much more
        accurate relevance scores than bi-encoders, but at O(n) cost per query.

        By re-scoring search results with a cross-encoder, we quantify the exact
        **"embedding gap"** per adversarial category:

        - **NDCG Improvement**: How much better results would be with perfect reranking
        - **Kendall's Tau**: Rank correlation between original and optimal ordering
        - **Gap Severity**: Classification of how much the bi-encoder approximation costs

        ### Interpretation
        - High NDCG improvement = the search engine found relevant documents but ranked
          them poorly (embedding gap is large)
        - Low Kendall's tau = the ranking is very different from optimal

        ### How to run
        ```bash
        searchprobe validate --run-id latest --cross-encoder ms-marco-MiniLM-L-12-v2
        ```
        """)
