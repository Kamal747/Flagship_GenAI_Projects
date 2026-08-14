"""
Chat UI: conversational natural-language analysis, backed by the Groq
tool-calling engine. Renders charts/tables inline as real tool outputs.

All new messages are appended to session state and the app is rerun rather
than displayed manually mid-script — this keeps message order consistent
(chat_input always renders below the latest exchange, not above it) since
Streamlit's sticky-bottom input positioning is unreliable inside tabs.
"""
from __future__ import annotations

import streamlit as st

from app.core import llm_engine, profiling
from app.core.config import validate_config
from app.core.df_utils import dedupe_columns


def render_chat_view() -> None:
    df = st.session_state.get("df")
    if df is None:
        st.info("👈 Upload a dataset from the sidebar to start chatting with your data.")
        return

    problems = validate_config()
    if problems:
        for p in problems:
            st.error(p)
        return

    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("qa_history", [])
    st.session_state.setdefault("dashboard_charts", [])

    for turn in st.session_state["chat_history"]:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            for chart_index, chart in enumerate(turn.get("charts", [])):
                st.plotly_chart(
                    chart,
                    width="stretch",
                    key=f"chat_{turn['role']}_{chart_index}_{id(turn)}"
            )
            for table in turn.get("tables", []):
                st.dataframe(dedupe_columns(table), width="stretch")
            if turn.get("tool_trace"):
                with st.expander("🔍 Tool execution trace (for transparency)"):
                    for step in turn["tool_trace"]:
                        st.code(
                            f"Tool: {step['tool']}\nArgs: {step['arguments']}\n\n{step['result_preview']}",
                            language="text",
                        )

    user_input = st.chat_input("Ask a question about your data...")
    if not user_input:
        return

    st.session_state["chat_history"].append({"role": "user", "content": user_input})

    with st.spinner("Analyzing your data..."):
        try:
            profile_summary = profiling.full_profile_summary(df)
            llm_messages = [
                {"role": t["role"], "content": t["content"]}
                for t in st.session_state["chat_history"][:-1]
                if t["role"] in ("user", "assistant")
            ]
            result = llm_engine.run_chat_turn(
                conversation_history=llm_messages,
                user_message=user_input,
                df=df,
                profile_summary=profile_summary,
            )
        except llm_engine.LLMEngineError as e:
            st.session_state["chat_history"].append(
                {"role": "assistant", "content": f"❌ {e}", "charts": [], "tables": []}
            )
            st.rerun()
            return

    st.session_state["chat_history"].append(
        {
            "role": "assistant",
            "content": result.reply_text,
            "charts": result.charts,
            "tables": result.tables,
            "tool_trace": result.tool_trace,
        }
    )
    st.session_state["qa_history"].append({"question": user_input, "answer": result.reply_text})

    # Feed any charts generated this turn into the Dashboard tab, Power-BI style.
    for chart in result.charts:
        chart_title = None
        if chart.layout.title and chart.layout.title.text:
            chart_title = chart.layout.title.text
        if not chart_title:
            # Defense in depth: charts.build_chart always sets a real title,
            # but if it's ever missing, fall back to the chart_type we
            # stamped into figure metadata rather than a meaningless
            # "Chart N" label with no indication of what it actually is.
            meta = chart.layout.meta or {}
            chart_type = meta.get("chart_type") if isinstance(meta, dict) else None
            chart_title = (chart_type or "chart").replace("_", " ").title()
        st.session_state["dashboard_charts"].append(
            {"title": chart_title, "figure": chart, "question": user_input}
        )

    st.rerun()
