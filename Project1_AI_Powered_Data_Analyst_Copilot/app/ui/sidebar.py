"""
Sidebar UI: file upload, sheet selection, dataset profile, and cleaning
suggestions. Pure Streamlit + calls into deterministic core modules.
"""
from __future__ import annotations

import streamlit as st

from app.core import cleaning, data_handler, profiling
from app.core.config import settings


def render_sidebar() -> None:
    st.sidebar.header("📁 Dataset")

    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV or Excel", type=["csv", "xlsx", "xls"], accept_multiple_files=False
    )

    if uploaded_file is not None:
        # Only reload if it's a new file
        if st.session_state.get("_last_uploaded_name") != uploaded_file.name:
            try:
                sheets = data_handler.load_file(uploaded_file, settings.max_upload_mb)
            except data_handler.DataLoadError as e:
                st.sidebar.error(f"❌ {e}")
                return

            st.session_state["_last_uploaded_name"] = uploaded_file.name
            st.session_state["sheets"] = sheets
            st.session_state["active_sheet"] = list(sheets.keys())[0]
            st.session_state["df"] = sheets[st.session_state["active_sheet"]].df
            st.session_state["cleaning_log"] = []
            st.session_state["chat_history"] = []
            st.session_state["qa_history"] = []

    sheets = st.session_state.get("sheets")
    if not sheets:
        st.sidebar.info("Upload a CSV or Excel file to get started.")
        return

    if len(sheets) > 1:
        selected_sheet = st.sidebar.selectbox("Sheet", list(sheets.keys()))
        if selected_sheet != st.session_state.get("active_sheet"):
            st.session_state["active_sheet"] = selected_sheet
            st.session_state["df"] = sheets[selected_sheet].df
            st.session_state["cleaning_log"] = []
            st.session_state["chat_history"] = []
            st.session_state["qa_history"] = []

    df = st.session_state["df"]

    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Quick Stats")
    shape = profiling.basic_shape(df)
    c1, c2 = st.sidebar.columns(2)
    c1.metric("Rows", f"{shape['rows']:,}")
    c2.metric("Columns", shape["columns"])
    c1.metric("Duplicates", shape["duplicate_rows"])
    c2.metric("Size (MB)", shape["memory_mb"])

    if shape["rows"] > settings.large_dataset_row_threshold:
        st.sidebar.warning(
            f"Large dataset ({shape['rows']:,} rows). Some operations may be slower."
        )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🧹 Cleaning Suggestions")
    suggestions = cleaning.generate_suggestions(df)

    if not suggestions:
        st.sidebar.success("No obvious data quality issues detected.")
    else:
        for s in suggestions:
            with st.sidebar.expander(f"{s.issue}" + (f" — {s.column}" if s.column else "")):
                st.write(s.detail)
                if st.button(s.action_label, key=f"apply_{s.id}"):
                    new_df = cleaning.apply_suggestion(df, s.id)
                    st.session_state["df"] = new_df
                    st.session_state.setdefault("cleaning_log", []).append(s.action_label)
                    st.rerun()

    if st.session_state.get("cleaning_log"):
        st.sidebar.markdown("**Applied this session:**")
        for action in st.session_state["cleaning_log"]:
            st.sidebar.caption(f"✅ {action}")

    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Reset session"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

