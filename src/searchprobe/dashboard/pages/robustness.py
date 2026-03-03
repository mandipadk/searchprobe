"""Dashboard page for perturbation robustness analysis."""

import streamlit as st

from searchprobe.reporting.charts import create_sensitivity_map_chart
from searchprobe.storage import Database


def render(db: Database, run_id: str) -> None:
    """Render the robustness analysis page."""
    st.header("Perturbation Robustness Analysis")
    st.markdown(
        "Measure search result stability under systematic query perturbations. "
        "Identifies which words are **load-bearing** for retrieval."
    )

    # Get perturbation results
    perturbation_results = db.get_perturbation_results(run_id)

    if not perturbation_results:
        st.info(
            "No perturbation results found for this run. "
            "Run `searchprobe perturb --run-id <id>` to analyze robustness."
        )
        _show_explanation()
        return

    # Summary metrics
    st.subheader("Overall Stability")
    import numpy as np

    jaccards = [r.get("jaccard_similarity", 0) for r in perturbation_results]
    rbos = [r.get("rbo_score", 0) for r in perturbation_results]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean Jaccard Similarity", f"{np.mean(jaccards):.3f}")
    with col2:
        st.metric("Mean RBO Score", f"{np.mean(rbos):.3f}")
    with col3:
        st.metric("Total Perturbations", len(perturbation_results))

    # Stability by operator
    st.subheader("Stability by Perturbation Type")
    operators: dict[str, list[float]] = {}
    for r in perturbation_results:
        op = r.get("operator", "unknown")
        if op not in operators:
            operators[op] = []
        operators[op].append(r.get("jaccard_similarity", 0))

    if operators:
        import plotly.graph_objects as go

        op_names = list(operators.keys())
        op_means = [float(np.mean(v)) for v in operators.values()]
        op_counts = [len(v) for v in operators.values()]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=[n.replace("_", " ").title() for n in op_names],
                    y=op_means,
                    text=[f"{m:.3f} (n={c})" for m, c in zip(op_means, op_counts)],
                    textposition="outside",
                    marker_color=["#e74c3c" if m < 0.5 else "#f1c40f" if m < 0.7 else "#2ecc71" for m in op_means],
                )
            ]
        )
        fig.update_layout(
            title="Mean Jaccard Similarity by Perturbation Type",
            yaxis_title="Jaccard Similarity",
            yaxis=dict(range=[0, 1.1]),
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Perturbation explorer
    st.subheader("Perturbation Explorer")
    for r in perturbation_results[:20]:
        jaccard = r.get("jaccard_similarity", 0)
        color = "red" if jaccard < 0.3 else "orange" if jaccard < 0.6 else "green"
        with st.expander(
            f"[{r.get('operator', '?')}] {r.get('original_query', '')[:50]}... "
            f"→ Jaccard: {jaccard:.3f}"
        ):
            st.markdown(f"**Original:** {r.get('original_query', '')}")
            st.markdown(f"**Perturbed:** {r.get('perturbed_query', '')}")
            st.markdown(f"**Jaccard:** :{color}[{jaccard:.4f}]")
            st.markdown(f"**RBO:** {r.get('rbo_score', 0):.4f}")


def _show_explanation() -> None:
    """Show explanation of perturbation analysis."""
    with st.expander("What is Perturbation Analysis?"):
        st.markdown("""
        Perturbation analysis systematically modifies queries and measures how much
        search results change. This reveals:

        - **Sensitivity Maps**: Which words are "load-bearing" for retrieval
        - **Stability Scores**: How consistent results are under small changes
        - **Provider Robustness**: Which providers are most/least stable

        ### Perturbation Operators
        - **Word Delete**: Remove one word at a time
        - **Word Swap**: Swap adjacent words
        - **Negation Insert/Remove**: Add or remove negation words
        - **Synonym Replace**: Replace words with synonyms

        ### How to run
        ```bash
        searchprobe perturb --run-id latest --operators word_delete,word_swap
        ```
        """)
