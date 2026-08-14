"""
Chart generation. Given a chart spec (type + columns), builds a Plotly figure
from the REAL dataframe (or a real aggregated result). No LLM-invented data.

Supports a broad Power BI / Tableau-style chart library covering essentially
every chart type commonly used for tabular business data: trend, comparison,
relationship, distribution, part-to-whole, hierarchical, flow, KPI, 3D,
geographic, and financial charts.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class ChartError(Exception):
    pass


SUPPORTED_TYPES = {
    # Trend
    "line", "area",
    # Comparison
    "bar", "bar_horizontal", "stacked_bar", "grouped_bar", "polar_bar",
    # Relationship
    "scatter", "bubble", "scatter_3d", "density_heatmap", "contour",
    # Part-to-whole
    "pie", "donut", "treemap", "sunburst",
    # Distribution
    "histogram", "box", "violin", "strip", "ecdf",
    # Correlation
    "heatmap",
    # Flow / change
    "funnel", "waterfall", "sankey",
    # Multi-metric / KPI
    "radar", "gauge", "bullet",
    # Financial
    "candlestick",
    # Geographic
    "choropleth",
}

# Chart types that consume a fixed, self-describing set of columns from the
# dataframe (via `data_code`) rather than the generic x/y/color/size — the
# LLM is instructed to name columns exactly as listed here.
_FIXED_SCHEMA_TYPES = {
    "sankey": ["source", "target", "value"],
    "candlestick": ["date", "open", "high", "low", "close"],
}


def _require_columns(df: pd.DataFrame, cols: list[str | None]) -> None:
    for col in filter(None, cols):
        if col not in df.columns:
            raise ChartError(f"Column '{col}' not found in the dataset. "
                              f"Available columns: {list(df.columns)}")


def _relevant_columns(chart_type: str, x, y, color, size, z) -> list[str | None]:
    """Only the columns actually consumed by this chart type should be
    validated — an LLM passing an unused/irrelevant column (e.g. 'color' on
    a plain pie chart) must not cause a spurious failure."""
    always_relevant = {
        "line": [x, y, color], "area": [x, y, color],
        "bar": [x, y, color], "bar_horizontal": [x, y, color],
        "grouped_bar": [x, y, color], "stacked_bar": [x, y, color],
        "polar_bar": [x, y, color],
        "scatter": [x, y, color, size], "bubble": [x, y, color, size],
        "scatter_3d": [x, y, z, color, size],
        "density_heatmap": [x, y],
        "contour": [x, y],
        "pie": [x, y], "donut": [x, y],
        "histogram": [x or y, color],
        "box": [x, y, color], "violin": [x, y, color], "strip": [x, y, color],
        "ecdf": [x or y, color],
        "heatmap": [],
        "funnel": [x, y, color],
        "waterfall": [x, y],
        "radar": [x, y],
        "gauge": [y],
        "bullet": [x, y],
        "choropleth": [x, y],
    }
    return always_relevant.get(chart_type, [x, y, color, size])


def _default_chart_title(chart_type: str, x, y, path) -> str:
    """Every chart must clearly show what it is, even if the caller (LLM)
    forgot to pass a title. Always incorporates the chart type name so a
    chart is identifiable at a glance, e.g. in the Dashboard tab."""
    label = chart_type.replace("_", " ").title()
    if chart_type in ("treemap", "sunburst") and path:
        return f"{label}: {y or 'Count'} by {' > '.join(path)}"
    if chart_type == "heatmap":
        return "Correlation Heatmap"
    if chart_type == "sankey":
        return "Sankey Diagram: Flow Breakdown"
    if chart_type == "candlestick":
        return "Candlestick Chart"
    if chart_type == "choropleth" and x and y:
        return f"{label}: {y} by {x}"
    if x and y:
        return f"{label}: {y} by {x}"
    if x:
        return f"{label}: {x}"
    if y:
        return f"{label}: {y}"
    return label


def build_chart(
    df: pd.DataFrame,
    chart_type: str,
    x: str | None = None,
    y: str | None = None,
    color: str | None = None,
    size: str | None = None,
    z: str | None = None,
    path: list[str] | None = None,
    title: str | None = None,
) -> go.Figure:
    """
    x, y, color, size, z: column names (z is only used by scatter_3d).
    path: for treemap/sunburst — an ordered list of columns forming the hierarchy
          (e.g. ["region", "product"]).
    sankey / candlestick use a fixed column schema instead of x/y — see
    _FIXED_SCHEMA_TYPES. Provide those exact column names via `data_code`.
    """
    chart_type = chart_type.lower().strip()
    if chart_type not in SUPPORTED_TYPES:
        raise ChartError(f"Unsupported chart type '{chart_type}'. Supported: {sorted(SUPPORTED_TYPES)}")

    # Always have a real, descriptive title — never blank, never generic.
    # Computed up front so it's available to every branch below (both the
    # plotly.express calls, which take `title=...`, and the manual
    # go.Figure() constructions, which set it via update_layout afterward).
    title = title.strip() if title and title.strip() else _default_chart_title(chart_type, x, y, path)

    chart_type_labels = {
    "line": "Line chart",
    "area": "Area chart",
    "bar": "Bar chart",
    "bar_horizontal": "Horizontal bar chart",
    "grouped_bar": "Grouped bar chart",
    "stacked_bar": "Stacked bar chart",
    "scatter": "Scatter plot",
    "bubble": "Bubble chart",
    "pie": "Pie chart",
    "donut": "Donut chart",
    "treemap": "Treemap",
    "sunburst": "Sunburst chart",
    "histogram": "Histogram",
    "box": "Box plot",
    "violin": "Violin plot",
    "funnel": "Funnel chart",
    "waterfall": "Waterfall chart",
    "radar": "Radar chart",
    "gauge": "Gauge chart",
    "heatmap": "Heatmap",
    "scatter_3d": "3D scatter plot",
    "density_heatmap": "Density heatmap",
    "contour": "Contour plot",
    "strip": "Strip plot",
    "ecdf": "ECDF plot",
    "polar_bar": "Polar bar chart",
    "sankey": "Sankey diagram",
    "bullet": "Bullet chart",
    "candlestick": "Candlestick chart",
    "choropleth": "Choropleth map",
    }

    label = chart_type_labels.get(chart_type, chart_type.title())

    if not title.lower().startswith(label.lower()):
        title = f"{label} – {title}"

    if chart_type in ("treemap", "sunburst"):
        _require_columns(df, [y, color])
    elif chart_type in _FIXED_SCHEMA_TYPES:
        _require_columns(df, _FIXED_SCHEMA_TYPES[chart_type])
    else:
        _require_columns(df, _relevant_columns(chart_type, x, y, color, size, z))
    if path:
        _require_columns(df, path)

    # Pie/donut/funnel are commonly asked for as "chart of counts by X" with
    # no explicit value column — fall back to counting occurrences of x.
    if chart_type in ("pie", "donut", "funnel") and x and not y:
        counted = df[x].value_counts().reset_index()
        counted.columns = [x, "count"]
        df, y = counted, "count"

    try:
        if chart_type == "line":
            fig = px.line(df, x=x, y=y, color=color, title=title, markers=True)

        elif chart_type == "area":
            fig = px.area(df, x=x, y=y, color=color, title=title)

        elif chart_type == "bar":
            fig = px.bar(df, x=x, y=y, color=color, title=title)

        elif chart_type == "bar_horizontal":
            # Robust to either argument order: some callers (LLM tool calls)
            # naturally think "x = horizontal axis = value" for a horizontal
            # bar, others follow the usual "x = category" convention used by
            # every other chart type. Detect numeric vs categorical instead
            # of blindly swapping — a wrong assumption here previously
            # produced a broken chart (categories forced onto the value axis).
            value_col, category_col = x, y
            if pd.api.types.is_numeric_dtype(df[x]) and not pd.api.types.is_numeric_dtype(df[y]):
                value_col, category_col = x, y
            elif pd.api.types.is_numeric_dtype(df[y]) and not pd.api.types.is_numeric_dtype(df[x]):
                value_col, category_col = y, x
            else:
                # Ambiguous (both/neither numeric) — fall back to the
                # standard "x = category, y = value" convention.
                value_col, category_col = y, x
            fig = px.bar(df, x=value_col, y=category_col, color=color, title=title, orientation="h")

        elif chart_type == "grouped_bar":
            fig = px.bar(df, x=x, y=y, color=color, title=title, barmode="group")

        elif chart_type == "stacked_bar":
            fig = px.bar(df, x=x, y=y, color=color, title=title, barmode="stack")

        elif chart_type == "polar_bar":
            if x is None or y is None:
                raise ChartError("Polar bar requires 'x' (categories) and 'y' (values).")
            fig = px.bar_polar(df, theta=x, r=y, color=color, title=title)

        elif chart_type == "scatter":
            fig = px.scatter(df, x=x, y=y, color=color, size=size, title=title)

        elif chart_type == "bubble":
            if not size:
                raise ChartError("Bubble charts require a 'size' column.")
            fig = px.scatter(df, x=x, y=y, color=color, size=size, title=title,
                              size_max=45)

        elif chart_type == "scatter_3d":
            if not z:
                raise ChartError("3D scatter requires a 'z' column in addition to 'x' and 'y'.")
            fig = px.scatter_3d(df, x=x, y=y, z=z, color=color, size=size, title=title)

        elif chart_type == "density_heatmap":
            fig = px.density_heatmap(df, x=x, y=y, title=title, color_continuous_scale="Blues")

        elif chart_type == "contour":
            fig = px.density_contour(df, x=x, y=y, color=color, title=title)

        elif chart_type == "pie":
            if not x or not y:
                raise ChartError("Pie chart requires at least a category column ('x').")
            fig = px.pie(df, names=x, values=y, title=title)

        elif chart_type == "donut":
            if not x or not y:
                raise ChartError("Donut chart requires at least a category column ('x').")
            fig = px.pie(df, names=x, values=y, title=title, hole=0.5)

        elif chart_type == "histogram":
            fig = px.histogram(df, x=x or y, color=color, title=title)

        elif chart_type == "box":
            fig = px.box(df, x=x, y=y, color=color, title=title)

        elif chart_type == "violin":
            fig = px.violin(df, x=x, y=y, color=color, box=True, points="outliers", title=title)

        elif chart_type == "strip":
            fig = px.strip(df, x=x, y=y, color=color, title=title)

        elif chart_type == "ecdf":
            fig = px.ecdf(df, x=x or y, color=color, title=title)

        elif chart_type == "heatmap":
            numeric_df = df.select_dtypes(include="number")
            if numeric_df.shape[1] < 2:
                raise ChartError("Heatmap needs at least 2 numeric columns.")
            corr = numeric_df.corr(numeric_only=True)
            fig = px.imshow(corr, text_auto=True, title=title or "Correlation Heatmap",
                             color_continuous_scale="RdBu_r", zmin=-1, zmax=1)

        elif chart_type == "treemap":
            if not path:
                raise ChartError("Treemap requires a 'path' (list of hierarchy columns).")
            fig = px.treemap(df, path=path, values=y, color=color, title=title)

        elif chart_type == "sunburst":
            if not path:
                raise ChartError("Sunburst requires a 'path' (list of hierarchy columns).")
            fig = px.sunburst(df, path=path, values=y, color=color, title=title)

        elif chart_type == "funnel":
            if not x or not y:
                raise ChartError("Funnel chart requires stage ('x') and value ('y') columns.")
            fig = px.funnel(df, x=y, y=x, color=color, title=title)

        elif chart_type == "waterfall":
            if x is None or y is None:
                raise ChartError("Waterfall requires 'x' (categories) and 'y' (values).")
            fig = go.Figure(go.Waterfall(
                x=df[x].tolist(), y=df[y].tolist(),
                connector={"line": {"color": "rgb(150,150,150)"}},
            ))
            fig.update_layout(title=title or "Waterfall Chart")

        elif chart_type == "sankey":
            cols = _FIXED_SCHEMA_TYPES["sankey"]
            src, tgt, val = df[cols[0]], df[cols[1]], df[cols[2]]
            labels = sorted(set(src) | set(tgt))
            index = {label: i for i, label in enumerate(labels)}
            fig = go.Figure(go.Sankey(
                node=dict(label=labels, pad=15, thickness=18,
                          line=dict(color="#6B7280", width=0.5)),
                link=dict(
                    source=[index[s] for s in src],
                    target=[index[t] for t in tgt],
                    value=list(val),
                ),
            ))
            fig.update_layout(title=title or "Sankey Diagram")

        elif chart_type == "radar":
            if x is None or y is None:
                raise ChartError("Radar chart requires 'x' (categories) and 'y' (values).")
            categories = df[x].tolist()
            values = df[y].tolist()
            fig = go.Figure(go.Scatterpolar(r=values, theta=categories, fill="toself"))
            fig.update_layout(title=title or "Radar Chart", polar=dict(radialaxis=dict(visible=True)))

        elif chart_type == "gauge":
            if y is None:
                raise ChartError("Gauge requires a 'y' column (or a single computed value).")
            value = float(df[y].iloc[0]) if len(df) else 0.0
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=value,
                title={"text": title},
                gauge={"axis": {"range": [None, max(value * 1.5, 1)]}},
            ))
            fig.update_layout(title=title)  # also set at figure level for dashboard display

        elif chart_type == "bullet":
            if x is None or y is None:
                raise ChartError("Bullet chart requires 'x' (label) and 'y' (value) columns.")
            value = float(df[y].iloc[0]) if len(df) else 0.0
            fig = go.Figure(go.Indicator(
                mode="number+gauge", value=value,
                title={"text": title},
                gauge={"shape": "bullet", "axis": {"range": [None, max(value * 1.5, 1)]}},
                domain={"x": [0.1, 1], "y": [0.2, 0.8]},
            ))
            fig.update_layout(height=180, title=title)  # also set at figure level for dashboard display

        elif chart_type == "candlestick":
            cols = _FIXED_SCHEMA_TYPES["candlestick"]
            fig = go.Figure(go.Candlestick(
                x=df[cols[0]], open=df[cols[1]], high=df[cols[2]],
                low=df[cols[3]], close=df[cols[4]],
            ))
            fig.update_layout(title=title or "Candlestick Chart")

        elif chart_type == "choropleth":
            if x is None or y is None:
                raise ChartError("Choropleth requires 'x' (country/region names) and 'y' (values).")
            fig = px.choropleth(
                df, locations=x, locationmode="country names", color=y,
                title=title, color_continuous_scale="Blues",
            )

        else:  # pragma: no cover
            raise ChartError("Unhandled chart type.")

    except ChartError:
        raise
    except Exception as e:  # noqa: BLE001
        raise ChartError(f"Failed to build chart: {e}")

    _apply_readable_styling(fig, chart_type, has_color_split=bool(color))
    fig.update_layout(meta={"chart_type": chart_type})  # for dashboard/debugging use
    return fig


# Strong, readable palette — deliberately darker/more saturated than Plotly's
# pastel defaults, since charts sit on a white card inside a dark-themed app
# and need to stay legible and high-contrast.
_DARK_TEXT = "#111827"
_GRID_COLOR = "#D1D5DB"
_AXIS_LINE_COLOR = "#6B7280"
_ACCENT_COLORWAY = [
    "#1D4ED8", "#B91C1C", "#0F766E", "#7C3AED",
    "#B45309", "#0369A1", "#4D7C0F", "#BE185D",
]

# Chart types whose primary trace(s) support marker_color / line_color overrides.
_TRACE_COLOR_COMPATIBLE = {
    "line", "area", "bar", "bar_horizontal", "grouped_bar", "stacked_bar",
    "polar_bar", "scatter", "bubble", "scatter_3d", "funnel", "waterfall",
    "strip", "ecdf",
}

# Chart types with no cartesian x/y axes — update_xaxes/update_yaxes are
# harmless no-ops on these, but skipped for clarity.
_NON_CARTESIAN_TYPES = {"pie", "donut", "sunburst", "treemap", "gauge", "bullet",
                         "sankey", "radar", "polar_bar", "choropleth"}


def _apply_readable_styling(fig: go.Figure, chart_type: str, has_color_split: bool) -> None:
    """
    Charts always render on a WHITE plot area with DARK, high-contrast text
    and a saturated accent color — regardless of the surrounding app theme
    (which may be dark) — so titles, axis labels, tick numbers, and data
    series all stay clearly legible.
    """
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color=_DARK_TEXT, size=13),
        title_font=dict(color=_DARK_TEXT, size=17),
        legend=dict(font=dict(color=_DARK_TEXT, size=12)),
        colorway=_ACCENT_COLORWAY,
    )
    if chart_type not in _NON_CARTESIAN_TYPES:
        axis_style = dict(
            color=_DARK_TEXT,
            tickfont=dict(color=_DARK_TEXT, size=12),
            title_font=dict(color=_DARK_TEXT, size=13),
            linecolor=_AXIS_LINE_COLOR,
            gridcolor=_GRID_COLOR,
            zerolinecolor=_AXIS_LINE_COLOR,
        )
        try:
            fig.update_xaxes(**axis_style)
            fig.update_yaxes(**axis_style)
        except Exception:  # noqa: BLE001 - e.g. 3D scenes use a different API
            pass

    # Single-series charts (no color grouping) default to Plotly's pale
    # pastel blue, which is hard to read — force the strong accent instead.
    if not has_color_split and chart_type in _TRACE_COLOR_COMPATIBLE:
        try:
            fig.update_traces(
                marker_color=_ACCENT_COLORWAY[0],
                line_color=_ACCENT_COLORWAY[0],
                selector=dict(type="scatter"),
            )
            fig.update_traces(marker_color=_ACCENT_COLORWAY[0], selector=dict(type="bar"))
            fig.update_traces(marker_color=_ACCENT_COLORWAY[0], selector=dict(type="scatter3d"))
            fig.update_traces(marker_color=_ACCENT_COLORWAY[0], selector=dict(type="scatterpolar"))
            fig.update_traces(line=dict(width=3), marker=dict(size=8), selector=dict(mode="lines+markers"))
        except Exception:  # noqa: BLE001 - styling is best-effort, never fatal
            pass
