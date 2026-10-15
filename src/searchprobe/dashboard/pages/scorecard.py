"""Scorecard page: per-document metrics from cached judgments (no LLM calls)."""

import asyncio

import pandas as pd
import streamlit as st

from searchprobe.storage import Database


@st.cache_data(show_spinner="Computing scorecard from cached judgments...")
def _compute(db_path: str, run_id: str, k: int):
    from searchprobe.config import get_settings
    from searchprobe.evidence import EvidenceFetcher, EvidenceStore
    from searchprobe.judging.engine import JudgingEngine
    from searchprobe.judging.llm_judge import DocumentJudge
    from searchprobe.judging.store import JudgmentStore
    from searchprobe.metrics.render import failure_exhibits
    from searchprobe.metrics.scorecard import build_scorecard, compute_list_metrics

    settings = get_settings()
    engine = JudgingEngine(
        Database(db_path),
        JudgmentStore(db_path),
        DocumentJudge(None, settings.judge_model, settings.judge_effort),
        fetcher=EvidenceFetcher(EvidenceStore(db_path), offline=True, max_age_days=None),
        allow_llm=False,
    )
    run = asyncio.run(engine.judge_run(run_id, k=k))
    rows = compute_list_metrics(run, k=k)
    return build_scorecard(rows).to_dict(), [r.to_dict() for r in rows], failure_exhibits(run, k)


def render(db: Database, run_id: str) -> None:
    st.markdown('<h1 class="main-header">🎯 Scorecard</h1>', unsafe_allow_html=True)
    st.caption(
        "Per-document judgments: violation@k is the share of the top-k results that break a "
        "hard constraint of the query. Brackets are 95% bootstrap CIs over cases. Populate with "
        "`searchprobe judge`."
    )
    k = st.sidebar.slider("k", 1, 10, 10)
    card, rows, exhibits = _compute(str(db.db_path), run_id, k)
    if not rows or not any(r["n_judged"] for r in rows):
        st.info("No judgments for this run yet. Run `searchprobe judge --run-id ...`.")
        return

    table = []
    for system, metrics in card["by_system"].items():
        row = {"System": system}
        for m, c in metrics.items():
            row[m] = f"{c['mean']:.3f} [{c['lower']:.3f}, {c['upper']:.3f}] (n={c['n']})"
        row["error rate"] = f"{card['error_rate'].get(system, 0):.0%}"
        table.append(row)
    st.subheader("Overall")
    st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)

    st.subheader("Violation@k by category")
    cat_rows = []
    for cat, systems in card["by_category"].items():
        r = {"Category": cat}
        for system, metrics in systems.items():
            r[system] = metrics["violation_at_k"]["mean"]
        cat_rows.append(r)
    st.dataframe(
        pd.DataFrame(cat_rows).set_index("Category").style.format("{:.0%}", na_rep="-"),
        use_container_width=True,
    )

    sig = [c for c in card["comparisons"] if c["significant"]]
    if sig:
        st.subheader("Significant paired differences (BH-corrected)")
        st.dataframe(pd.DataFrame(sig), use_container_width=True, hide_index=True)

    if exhibits:
        st.subheader("Failure exhibits")
        for e in exhibits:
            with st.expander(f"{e['system']} · rank {e['rank']} · {e['query']}"):
                st.write(e["url"])
                st.write(f"**Violates:** {e['constraint']}")
                if e["evidence"]:
                    st.write(f"**Evidence ({e['source']}):** “{e['evidence']}”")
