"""Streamlit dashboard entry point."""

import streamlit as st
from pathlib import Path

from searchprobe.config import get_settings
from searchprobe.storage import Database


def get_db() -> Database:
    """Get cached database connection."""
    settings = get_settings()
    return Database(settings.database_path)


def run_dashboard() -> None:
    """Run the Streamlit dashboard."""
    st.set_page_config(
        page_title="SearchProbe Dashboard",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1a1a2e;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar navigation
    st.sidebar.title("🔍 SearchProbe")
    st.sidebar.markdown("---")

    # Page selection
    page = st.sidebar.radio(
        "Navigate",
        [
            "📊 Overview",
            "🏆 Provider Comparison",
            "📁 Category Deep Dive",
            "🔎 Query Explorer",
            "⚠️ Failure Analysis",
            "🧬 Embedding Geometry",
            "🔧 Robustness Analysis",
            "✅ Cross-Encoder Validation",
        ],
    )

    # Run selection
    db = get_db()
    runs = _get_available_runs(db)

    if not runs:
        st.warning("No benchmark runs found. Run `searchprobe run` first.")
        return

    selected_run = st.sidebar.selectbox(
        "Select Run",
        runs,
        format_func=lambda x: f"{x['name'] or 'Unnamed'} ({x['id'][:8]}...)",
    )

    st.sidebar.markdown("---")
    st.sidebar.info(f"**Run ID:** `{selected_run['id'][:8]}...`\n\n**Started:** {selected_run['started_at'][:19] if selected_run.get('started_at') else 'N/A'}")

    # Route to page
    if page == "📊 Overview":
        from searchprobe.dashboard.pages.overview import render
        render(db, selected_run["id"])
    elif page == "🏆 Provider Comparison":
        from searchprobe.dashboard.pages.provider_comparison import render
        render(db, selected_run["id"])
    elif page == "📁 Category Deep Dive":
        from searchprobe.dashboard.pages.category_deep_dive import render
        render(db, selected_run["id"])
    elif page == "🔎 Query Explorer":
        from searchprobe.dashboard.pages.query_explorer import render
        render(db, selected_run["id"])
    elif page == "⚠️ Failure Analysis":
        from searchprobe.dashboard.pages.failure_analysis import render
        render(db, selected_run["id"])
    elif page == "🧬 Embedding Geometry":
        from searchprobe.dashboard.pages.embedding_geometry import render
        render(db, selected_run["id"])
    elif page == "🔧 Robustness Analysis":
        from searchprobe.dashboard.pages.robustness import render
        render(db, selected_run["id"])
    elif page == "✅ Cross-Encoder Validation":
        from searchprobe.dashboard.pages.validation import render
        render(db, selected_run["id"])


def _get_available_runs(db: Database) -> list[dict]:
    """Get list of available benchmark runs."""
    with db._get_connection() as conn:
        rows = conn.execute(
            """SELECT id, name, started_at, completed_at, cost_total
               FROM runs ORDER BY started_at DESC LIMIT 50"""
        ).fetchall()
    return [dict(row) for row in rows]


if __name__ == "__main__":
    run_dashboard()
