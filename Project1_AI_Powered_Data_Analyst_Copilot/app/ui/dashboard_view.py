"""
Dashboard tab: every chart generated in the chat session, laid out in a
Power-BI-style grid, all in one place for a quick visual overview.
"""
from __future__ import annotations

import streamlit as st

DASHBOARD_COLUMNS = 2


def render_dashboard_view() -> None:
    dashboard_charts = st.session_state.get("dashboard_charts", [])

    if not dashboard_charts:
        st.info(
            "📊 No charts yet. Ask for a chart in the **Chat** tab (e.g. "
            "\"Show me a bar chart of revenue by region\") and it will "
            "automatically appear here."
        )
        return

    col1, col2 = st.columns([4, 1])
    with col1:
        st.caption(f"{len(dashboard_charts)} chart(s) generated this session — laid out like a BI dashboard.")
    with col2:
        if st.button("🗑️ Clear dashboard", width="stretch"):
            st.session_state["dashboard_charts"] = []
            st.rerun()

    st.markdown("---")

    columns = st.columns(DASHBOARD_COLUMNS)
    for idx, item in enumerate(dashboard_charts):
        col = columns[idx % DASHBOARD_COLUMNS]
        with col:
            with st.container(border=True):
                st.markdown(f"**{item['title']}**")
                st.plotly_chart(
                    item["figure"],
                    width="stretch",
                    key=f"dashboard_chart_{idx}",
                )

