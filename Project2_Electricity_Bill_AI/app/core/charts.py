"""
Chart generation for the old-policy-vs-new-policy bill comparison dashboard.
All chart data comes directly from the deterministic BillResult/
ComparisonResult objects — nothing here invents or estimates numbers, it
only visualizes them. Charts always render on a white background with dark,
readable text.
"""
from __future__ import annotations

import plotly.graph_objects as go

from app.core.calculator import BillResult
from app.core.comparison import ComparisonResult

_DARK_TEXT = "#111827"
_OLD_COLOR = "#6B7280"
_NEW_COLOR = "#1D4ED8"


def _style(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(color=_DARK_TEXT, size=16)),
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color=_DARK_TEXT, size=12),
        legend=dict(font=dict(color=_DARK_TEXT)),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    fig.update_xaxes(color=_DARK_TEXT, tickfont=dict(color=_DARK_TEXT), gridcolor="#E5E7EB")
    fig.update_yaxes(color=_DARK_TEXT, tickfont=dict(color=_DARK_TEXT), gridcolor="#E5E7EB")
    return fig


def create_free_units_chart(old_bill: BillResult, new_bill: BillResult) -> go.Figure:
    """Free-unit subsidy comparison — the core driver of the policy
    difference (e.g. old policy 100 free units always vs new policy 200
    free units when total consumption qualifies)."""
    old_free = old_bill.slab_breakdown[0].units_in_slab if old_bill.slab_breakdown[0].rate == 0 else 0
    new_free = new_bill.slab_breakdown[0].units_in_slab if new_bill.slab_breakdown[0].rate == 0 else 0
    fig = go.Figure(go.Bar(
        x=["Old Policy", "New Policy"],
        y=[old_free, new_free],
        marker_color=[_OLD_COLOR, _NEW_COLOR],
        text=[f"{old_free:g} units", f"{new_free:g} units"],
        textposition="outside",
    ))
    return _style(fig, "Old Policy vs New Policy — Free Units Subsidy")


def create_bill_comparison_chart(comparison: ComparisonResult) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=["Old Policy", "New Policy"],
        y=[comparison.previous_bill, comparison.current_bill],
        marker_color=[_OLD_COLOR, _NEW_COLOR],
        text=[f"₹{comparison.previous_bill}", f"₹{comparison.current_bill}"],
        textposition="outside",
    ))
    return _style(fig, "Old Policy vs New Policy — Total Electricity Bill")


def create_slab_units_chart(comparison: ComparisonResult) -> go.Figure:
    labels = [line.slab_label for line in comparison.slab_comparison]
    fig = go.Figure()
    fig.add_bar(name="Old Policy", x=labels, y=[line.previous_units for line in comparison.slab_comparison],
                marker_color=_OLD_COLOR)
    fig.add_bar(name="New Policy", x=labels, y=[line.current_units for line in comparison.slab_comparison],
                marker_color=_NEW_COLOR)
    fig.update_layout(barmode="group")
    return _style(fig, "Slab-wise Consumption — Old Policy vs New Policy")


def create_slab_bill_chart(comparison: ComparisonResult) -> go.Figure:
    labels = [line.slab_label for line in comparison.slab_comparison]
    fig = go.Figure()
    fig.add_bar(name="Old Policy", x=labels, y=[line.previous_amount for line in comparison.slab_comparison],
                marker_color=_OLD_COLOR)
    fig.add_bar(name="New Policy", x=labels, y=[line.current_amount for line in comparison.slab_comparison],
                marker_color=_NEW_COLOR)
    fig.update_layout(barmode="group")
    return _style(fig, "Slab-wise Bill Amount — Old Policy vs New Policy")


def create_charts(comparison: ComparisonResult, old_bill: BillResult, new_bill: BillResult) -> dict[str, go.Figure]:
    """Returns all standard dashboard charts, keyed by a short identifier."""
    return {
        "units_comparison": create_free_units_chart(old_bill, new_bill),
        "bill_comparison": create_bill_comparison_chart(comparison),
        "slab_units_comparison": create_slab_units_chart(comparison),
        "slab_bill_comparison": create_slab_bill_chart(comparison),
    }
