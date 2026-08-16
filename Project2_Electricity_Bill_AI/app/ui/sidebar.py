"""
Sidebar UI: consumption input, tariff policy display, calculate/reset.
"""
from __future__ import annotations

import streamlit as st

from app.core.tariff import (
    OLD_POLICY_SLABS, NEW_POLICY_TIER_1_SLABS, NEW_POLICY_TIER_2_SLABS,
    NEW_POLICY_TIER_THRESHOLD_UNITS, FIXED_CHARGE_PER_MONTH,
    OLD_POLICY_NAME, NEW_POLICY_NAME,
)


def render_sidebar() -> None:
    st.sidebar.header("⚡ Bill Inputs")
    st.sidebar.caption(
        "Enter your consumption once — the app compares what it would cost "
        "under the **old tariff policy** vs the **new tariff policy**."
    )

    units = st.sidebar.number_input(
        "Units Consumed", min_value=0.0, value=st.session_state.get("units_input", 450.0),
        step=1.0, key="units_input",
    )

    col1, col2 = st.sidebar.columns(2)
    calculate_clicked = col1.button("⚡ Calculate", type="primary", width="stretch")
    reset_clicked = col2.button("🔄 Reset", width="stretch")

    if calculate_clicked:
        st.session_state["calculated"] = True
        st.session_state["units"] = units
        st.session_state["chat_history"] = []

    if reset_clicked:
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 Tariff Policy Configuration")
    st.sidebar.warning(
        "⚠️ Verify these rates against official TNERC/TANGEDCO tariff "
        "orders at tnerc.tn.gov.in before using this for real billing "
        "decisions. Edit `app/core/tariff.py` to update.",
        icon="⚠️",
    )
    with st.sidebar.expander(f"🔴 {OLD_POLICY_NAME}"):
        for slab in OLD_POLICY_SLABS:
            st.caption(f"{slab.label}: ₹{slab.rate}/unit")
    with st.sidebar.expander(f"🟢 {NEW_POLICY_NAME}"):
        st.caption(f"If total ≤{NEW_POLICY_TIER_THRESHOLD_UNITS} units:")
        for slab in NEW_POLICY_TIER_1_SLABS:
            st.caption(f"  {slab.label}: ₹{slab.rate}/unit")
        st.caption(f"If total >{NEW_POLICY_TIER_THRESHOLD_UNITS} units:")
        for slab in NEW_POLICY_TIER_2_SLABS:
            st.caption(f"  {slab.label}: ₹{slab.rate}/unit")
    if FIXED_CHARGE_PER_MONTH > 0:
        st.sidebar.caption(f"Fixed charge: ₹{FIXED_CHARGE_PER_MONTH}/month")
