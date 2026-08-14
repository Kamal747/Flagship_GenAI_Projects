"""
Main profiling view: data preview, full column profile, correlation heatmap.
"""
from __future__ import annotations

import streamlit as st

from app.core import profiling
from app.core.config import settings


def render_profiling_view() -> None:
    df = st.session_state.get("df")
    if df is None:
        st.info("👈 Upload a dataset from the sidebar to begin.")
        return

    st.subheader("Dataset Preview")
    st.dataframe(df.head(settings.max_rows_preview), width="stretch")

    st.subheader("Column Profile")
    st.dataframe(profiling.column_profile(df), width="stretch")

    corr = profiling.numeric_correlations(df)
    if corr is not None:
        st.subheader("Numeric Correlations")
        st.dataframe(corr, width="stretch")
    else:
        st.caption("Not enough numeric columns for a correlation matrix.")
