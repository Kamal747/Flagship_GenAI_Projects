"""
Main dashboard view: old-policy vs new-policy bill summaries, slab-wise
breakdowns, comparison table, and charts — all for the SAME consumption
value. All numbers come directly from calculator.py / comparison.py — no
LLM involvement.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app.core.calculator import calculate_bill, BillResult
from app.core.comparison import compare_bills, ComparisonResult
from app.core.charts import create_charts


def _slab_breakdown_df(bill: BillResult) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Slab": line.slab_label,
            "Units in Slab": line.units_in_slab,
            "Rate (₹/unit)": line.rate,
            "Amount (₹)": line.amount,
        }
        for line in bill.slab_breakdown
    ])


def _render_bill_summary(title: str, bill: BillResult) -> None:
    st.subheader(title)
    st.caption(f"Policy: **{bill.policy_name}**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Units", f"{bill.units:g}")
    c2.metric("Energy Charge", f"₹{bill.energy_charge:,.2f}")
    c3.metric("Total Bill", f"₹{bill.total_bill:,.2f}")
    st.dataframe(_slab_breakdown_df(bill), width="stretch", hide_index=True)


def _render_comparison_summary(comparison: ComparisonResult) -> None:
    st.subheader("📊 Old Policy vs New Policy Comparison")
    st.caption(f"Same consumption ({comparison.previous_units:g} units) under both policies:")
    c1, c2, c3 = st.columns(3)
    c1.metric("Units Consumed", f"{comparison.previous_units:g}")
    c2.metric("Old Policy Bill", f"₹{comparison.previous_bill:,.2f}")
    pct_label = f"{comparison.percentage_change:+.1f}%" if comparison.percentage_change is not None else "N/A"
    c3.metric("New Policy Bill", f"₹{comparison.current_bill:,.2f}",
              delta=f"₹{comparison.bill_difference:,.2f} ({pct_label})", delta_color="inverse")

    if comparison.biggest_increase_slab:
        st.info(f"📈 The **{comparison.biggest_increase_slab}** slab contributed the most to the bill increase.")
    if comparison.biggest_contributor_slab_current:
        st.caption(f"Biggest contributor under the new policy: **{comparison.biggest_contributor_slab_current}**.")
    if comparison.structure_changed:
        st.warning(
            f"⚠️ The old policy (**{comparison.previous_policy_name}**) and new policy "
            f"(**{comparison.current_policy_name}**) use different slab boundaries for this "
            f"consumption level, which is why the slab-wise table below shows more rows than usual.",
            icon="⚠️",
        )


def _render_slab_comparison_table(comparison: ComparisonResult) -> None:
    st.subheader("📋 Slab-wise Comparison")
    df = pd.DataFrame([
        {
            "Slab": line.slab_label,
            "Old Policy Units": line.previous_units,
            "Old Policy Amount (₹)": line.previous_amount,
            "New Policy Units": line.current_units,
            "New Policy Amount (₹)": line.current_amount,
            "Difference (₹)": line.amount_difference,
        }
        for line in comparison.slab_comparison
    ])
    st.dataframe(df, width="stretch", hide_index=True)


def render_dashboard() -> ComparisonResult | None:
    """Renders the full dashboard. Returns the ComparisonResult so the chat
    tab can reuse it as chatbot context, or None if not yet calculated."""
    if not st.session_state.get("calculated"):
        st.info("👈 Enter your units consumed in the sidebar, then click **Calculate**.")
        return None

    units = st.session_state["units"]

    try:
        old_bill = calculate_bill(units, policy="old")
        new_bill = calculate_bill(units, policy="new")
    except Exception as e:  # noqa: BLE001
        st.error(f"❌ Could not calculate bill: {e}")
        return None

    comparison = compare_bills(old_bill, new_bill)

    _render_comparison_summary(comparison)
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        _render_bill_summary("🔴 Old Policy Bill Summary", old_bill)
    with col2:
        _render_bill_summary("🟢 New Policy Bill Summary", new_bill)

    st.markdown("---")
    _render_slab_comparison_table(comparison)

    st.markdown("---")
    st.subheader("📈 Charts")
    charts = create_charts(comparison, old_bill, new_bill)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(charts["units_comparison"], width="stretch")
        st.plotly_chart(charts["slab_units_comparison"], width="stretch")
    with c2:
        st.plotly_chart(charts["bill_comparison"], width="stretch")
        st.plotly_chart(charts["slab_bill_comparison"], width="stretch")

    return comparison
