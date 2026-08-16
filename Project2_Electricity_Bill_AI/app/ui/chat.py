"""
Chat UI: natural-language Q&A about the already-calculated bill, backed by
the Groq chatbot in chatbot.py. The chatbot only narrates the structured
comparison summary — see chatbot.SYSTEM_PROMPT for the grounding rule.
"""
from __future__ import annotations

import streamlit as st

from app.core import chatbot
from app.core.comparison import ComparisonResult, generate_comparison_summary
from app.core.config import validate_config


def render_chat(comparison: ComparisonResult | None) -> None:
    st.subheader("🤖 Ask About Your Bill")

    if comparison is None:
        st.info("Calculate your bill first (see sidebar) to start chatting about it.")
        return

    problems = validate_config()
    if problems:
        for p in problems:
            st.error(p)
        return

    st.session_state.setdefault("chat_history", [])

    st.caption(
        "Try: \"What's the old policy bill?\", \"How many do I save in new policy?\", "
        "\"Which slab contributes most to my bill?\""
    )

    for turn in st.session_state["chat_history"]:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    user_input = st.chat_input("Ask about your electricity bill...")
    if not user_input:
        return

    st.session_state["chat_history"].append({"role": "user", "content": user_input})

    with st.spinner("Thinking..."):
        try:
            summary = generate_comparison_summary(comparison)
            history_for_llm = [
                {"role": t["role"], "content": t["content"]}
                for t in st.session_state["chat_history"][:-1]
            ]
            reply = chatbot.ask_chatbot(user_input, summary, history_for_llm)
        except chatbot.ChatbotError as e:
            reply = f"❌ {e}"

    st.session_state["chat_history"].append({"role": "assistant", "content": reply})
    st.rerun()
