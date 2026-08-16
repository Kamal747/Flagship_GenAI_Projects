"""
AI-Powered Electricity Bill Calculator & Comparison Chatbot — entrypoint.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.core.config import settings, validate_config
from app.ui.chat import render_chat
from app.ui.dashboard import render_dashboard
from app.ui.sidebar import render_sidebar

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)


def main() -> None:
    st.set_page_config(
        page_title=settings.app_title,
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title(f"⚡ {settings.app_title}")
    st.caption(
        "Enter your electricity consumption once to see what it would cost under the "
        "old tariff policy vs the new tariff policy — with an AI chatbot to explain the results."
    )

    for p in validate_config():
        st.warning(p)

    render_sidebar()
    comparison = render_dashboard()

    st.markdown("---")
    render_chat(comparison)


if __name__ == "__main__":
    main()