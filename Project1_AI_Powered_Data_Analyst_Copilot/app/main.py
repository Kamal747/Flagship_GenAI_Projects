"""
AI-Powered Data Analyst Copilot — main Streamlit entrypoint.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.core import report as report_module
from app.core.config import settings, validate_config
from app.ui.chat import render_chat_view
from app.ui.dashboard_view import render_dashboard_view
from app.ui.profiling_view import render_profiling_view
from app.ui.sidebar import render_sidebar

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main() -> None:
    st.set_page_config(
        page_title=settings.app_title,
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title(f"📊 {settings.app_title}")
    st.caption(
        "Upload a dataset, explore it, and ask questions in plain English. "
        "Every number you see is computed live from your real data — not guessed by the AI."
    )

    config_problems = validate_config()
    if config_problems:
        for p in config_problems:
            st.warning(p)

    render_sidebar()

    df = st.session_state.get("df")
    if df is None:
        st.markdown(
            """
            ### 👋 Get started
            1. Upload a CSV or Excel file using the sidebar.
            2. Review the automatic profile and cleaning suggestions.
            3. Ask questions in the **Chat** tab — e.g. *"What's the average revenue by region?"*
               or *"Show me a trend chart of sales over time."*
            4. Download a full analysis report when you're done.
            """
        )
        return

    tab_chat, tab_profile, tab_dashboard, tab_report = st.tabs(
        ["💬 Chat", "📋 Profile & Cleaning", "📊 Dashboard", "📄 Report"]
    )

    with tab_chat:
        render_chat_view()

    with tab_profile:
        render_profiling_view()

    with tab_dashboard:
        render_dashboard_view()

    with tab_report:
        st.subheader("Generate Analysis Report")
        st.caption(
            "Builds a Markdown report from your dataset profile, applied cleaning actions, "
            "and this session's Q&A — narrated by AI but grounded entirely in real computed results."
        )
        if st.button("📄 Generate Report"):
            with st.spinner("Generating report..."):
                sheet_name = st.session_state.get("active_sheet", "Sheet1")
                dataset_name = st.session_state.get("_last_uploaded_name", "dataset")
                report_md = report_module.build_report(
                    dataset_name=f"{dataset_name} ({sheet_name})",
                    df=df,
                    cleaning_log=st.session_state.get("cleaning_log", []),
                    qa_history=st.session_state.get("qa_history", []),
                )
                st.session_state["last_report"] = report_md

        if st.session_state.get("last_report"):
            st.markdown("---")
            st.markdown(st.session_state["last_report"])
            st.download_button(
                "⬇️ Download Report (Markdown)",
                data=st.session_state["last_report"],
                file_name="data_analysis_report.md",
                mime="text/markdown",
            )


if __name__ == "__main__":
    main()

