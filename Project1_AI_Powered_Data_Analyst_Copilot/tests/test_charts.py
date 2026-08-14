import pandas as pd
import pytest

from app.core import charts


@pytest.fixture
def grouped_df():
    return pd.DataFrame({
        "region": ["North", "South", "East", "West"],
        "revenue": [100.0, 200.0, 150.0, 175.0],
    })


@pytest.fixture
def hierarchy_df():
    return pd.DataFrame({
        "region": ["North", "North", "South", "South"],
        "product": ["A", "B", "A", "B"],
        "revenue": [50.0, 60.0, 70.0, 80.0],
    })


@pytest.mark.parametrize("chart_type", [
    "line", "area", "bar", "bar_horizontal", "grouped_bar", "stacked_bar", "polar_bar",
    "scatter", "pie", "donut", "histogram", "box", "violin", "strip", "ecdf",
    "funnel", "waterfall", "radar", "bullet",
])
def test_basic_chart_types_build(grouped_df, chart_type):
    fig = charts.build_chart(grouped_df, chart_type, x="region", y="revenue")
    assert fig is not None


def test_scatter_3d_builds():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    fig = charts.build_chart(df, "scatter_3d", x="a", y="b", z="c")
    assert fig is not None


def test_scatter_3d_requires_z():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    with pytest.raises(charts.ChartError):
        charts.build_chart(df, "scatter_3d", x="a", y="b")


def test_density_heatmap_and_contour_build():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 3, 2, 1]})
    assert charts.build_chart(df, "density_heatmap", x="a", y="b") is not None
    assert charts.build_chart(df, "contour", x="a", y="b") is not None


def test_sankey_builds_from_fixed_schema():
    df = pd.DataFrame({
        "source": ["A", "A", "B"],
        "target": ["B", "C", "C"],
        "value": [10, 20, 15],
    })
    fig = charts.build_chart(df, "sankey")
    assert fig is not None
    assert isinstance(fig.data[0], type(charts.go.Sankey()))


def test_sankey_requires_fixed_columns():
    df = pd.DataFrame({"a": [1], "b": [2]})
    with pytest.raises(charts.ChartError):
        charts.build_chart(df, "sankey")


def test_candlestick_builds_from_fixed_schema():
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=3),
        "open": [100, 102, 101], "high": [105, 106, 104],
        "low": [98, 100, 99], "close": [103, 101, 102],
    })
    fig = charts.build_chart(df, "candlestick")
    assert fig is not None


def test_choropleth_builds():
    df = pd.DataFrame({"country": ["United States", "India", "France"], "count": [980, 430, 90]})
    fig = charts.build_chart(df, "choropleth", x="country", y="count")
    assert fig is not None


def test_total_supported_chart_types_is_comprehensive():
    # Guards against accidental removal of chart types during future edits.
    assert len(charts.SUPPORTED_TYPES) >= 28


def test_default_title_used_when_none_given():
    df = pd.DataFrame({"region": ["N", "S"], "revenue": [1, 2]})
    fig = charts.build_chart(df, "bar", x="region", y="revenue")
    assert fig.layout.title.text == "Bar: revenue by region"


def test_default_title_for_heatmap_and_sankey():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    fig = charts.build_chart(df, "heatmap")
    assert fig.layout.title.text == "Correlation Heatmap"

    sankey_df = pd.DataFrame({"source": ["A"], "target": ["B"], "value": [10]})
    fig2 = charts.build_chart(sankey_df, "sankey")
    assert "Sankey" in fig2.layout.title.text


def test_gauge_and_bullet_title_set_at_figure_level():
    # Regression: go.Indicator sets title on the trace, not fig.layout.title,
    # by default — the Dashboard tab reads fig.layout.title.text, so this
    # must be explicitly populated too, or dashboard cards show no name.
    df = pd.DataFrame({"region": ["North"], "revenue": [500]})
    gauge_fig = charts.build_chart(df, "gauge", y="revenue")
    assert gauge_fig.layout.title.text and gauge_fig.layout.title.text != "None"

    bullet_fig = charts.build_chart(df, "bullet", x="region", y="revenue")
    assert bullet_fig.layout.title.text and bullet_fig.layout.title.text != "None"


def test_chart_type_stamped_into_meta_for_dashboard_fallback():
    df = pd.DataFrame({"region": ["N"], "revenue": [1]})
    fig = charts.build_chart(df, "bar", x="region", y="revenue")
    assert fig.layout.meta["chart_type"] == "bar"


def test_explicit_title_still_respected_when_given():
    df = pd.DataFrame({"region": ["N"], "revenue": [1]})
    fig = charts.build_chart(df, "bar", x="region", y="revenue", title="My Custom Title")
    assert fig.layout.title.text == "My Custom Title"


def test_bubble_requires_size(grouped_df):
    with pytest.raises(charts.ChartError):
        charts.build_chart(grouped_df, "bubble", x="region", y="revenue")

    fig = charts.build_chart(grouped_df, "bubble", x="region", y="revenue", size="revenue")
    assert fig is not None


def test_treemap_requires_path(hierarchy_df):
    with pytest.raises(charts.ChartError):
        charts.build_chart(hierarchy_df, "treemap", y="revenue")

    fig = charts.build_chart(hierarchy_df, "treemap", y="revenue", path=["region", "product"])
    assert fig is not None


def test_gauge_uses_single_value():
    df = pd.DataFrame({"kpi": [42.0]})
    fig = charts.build_chart(df, "gauge", y="kpi")
    assert fig is not None


def test_unsupported_chart_type_rejected(grouped_df):
    with pytest.raises(charts.ChartError):
        charts.build_chart(grouped_df, "not_a_real_chart", x="region", y="revenue")


def test_missing_column_rejected(grouped_df):
    with pytest.raises(charts.ChartError):
        charts.build_chart(grouped_df, "bar", x="nonexistent", y="revenue")


def test_pie_falls_back_to_value_counts_when_y_missing():
    df = pd.DataFrame({"category": ["a", "a", "b", "c", "c", "c"]})
    fig = charts.build_chart(df, "pie", x="category")
    assert fig is not None
    assert sum(fig.data[0].values) == 6


def test_funnel_falls_back_to_value_counts_when_y_missing():
    df = pd.DataFrame({"stage": ["viewed", "viewed", "clicked", "purchased"]})
    fig = charts.build_chart(df, "funnel", x="stage")
    assert fig is not None


def test_pie_ignores_irrelevant_unused_columns(grouped_df):
    # A color/size column that doesn't exist must NOT break a chart type
    # that never actually consumes it (this was the reported bug).
    fig = charts.build_chart(
        grouped_df, "pie", x="region", y="revenue",
        color="does_not_exist", size="also_missing",
    )
    assert fig is not None


def test_funnel_ignores_irrelevant_size_column(grouped_df):
    fig = charts.build_chart(grouped_df, "funnel", x="region", y="revenue", size="not_a_column")
    assert fig is not None


def test_charts_render_white_background_dark_text(grouped_df):
    fig = charts.build_chart(grouped_df, "bar", x="region", y="revenue")
    assert fig.layout.plot_bgcolor == "#FFFFFF"
    assert fig.layout.paper_bgcolor == "#FFFFFF"
    assert fig.layout.font.color == "#111827"


def test_single_series_line_chart_uses_strong_accent_color():
    df = pd.DataFrame({"year": [2019, 2020, 2021], "count": [10, 50, 30]})
    fig = charts.build_chart(df, "line", x="year", y="count")
    assert fig.data[0].line.color == "#1D4ED8"
    assert fig.data[0].line.width == 3


def test_axis_titles_and_ticks_are_dark(grouped_df):
    fig = charts.build_chart(grouped_df, "bar", x="region", y="revenue")
    assert fig.layout.xaxis.tickfont.color == "#111827"
    assert fig.layout.xaxis.title.font.color == "#111827"
    assert fig.layout.yaxis.tickfont.color == "#111827"
    assert fig.layout.yaxis.title.font.color == "#111827"


def test_bar_horizontal_correct_with_standard_arg_order(grouped_df):
    # x=category, y=value (the convention used by every other chart type)
    fig = charts.build_chart(grouped_df, "bar_horizontal", x="region", y="revenue")
    assert fig.data[0].orientation == "h"
    assert list(fig.data[0].x) == list(grouped_df["revenue"])
    assert list(fig.data[0].y) == list(grouped_df["region"])


def test_bar_horizontal_correct_with_reversed_arg_order(grouped_df):
    # x=value, y=category — some callers naturally think of "x" as the
    # horizontal (value) axis for a horizontal bar. Must still render right.
    fig = charts.build_chart(grouped_df, "bar_horizontal", x="revenue", y="region")
    assert fig.data[0].orientation == "h"
    assert list(fig.data[0].x) == list(grouped_df["revenue"])
    assert list(fig.data[0].y) == list(grouped_df["region"])

