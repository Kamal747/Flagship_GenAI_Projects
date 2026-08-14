"""
Tool (function) schemas exposed to the Groq LLM for tool-calling, plus the
dispatcher that routes a tool call to the correct deterministic module.
Every tool here executes real code/queries against the real dataframe.
"""
from __future__ import annotations

import json

import pandas as pd

from app.core import anomaly, charts, df_utils, profiling, sandbox, sql_engine

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_profile",
            "description": "Get deterministic structural profiling info about the dataset "
                            "(shape, column dtypes, missing %, unique counts, basic numeric stats).",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_pandas_code",
            "description": "Execute a short Pandas snippet against the real dataframe `df` to "
                            "compute an exact answer. Must assign the answer to a variable `result`. "
                            "Only pandas (pd) and numpy (np) are available.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Pandas code operating on `df`, ending with `result = ...`",
                    }
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": "Execute a read-only SQL SELECT query against a DuckDB table named "
                            "'dataset' that mirrors the real dataframe.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "A SELECT (or WITH) SQL query."}
                },
                "required": ["sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "build_chart",
            "description": "Create a chart. Supports a very broad, Power BI / Tableau-style "
                            "chart library covering trend, comparison, relationship, distribution, "
                            "part-to-whole, hierarchical, flow, KPI, 3D, geographic, and financial "
                            "charts. Choose the type that best fits the user's request: 'bubble'/"
                            "'scatter_3d' for extra dimensions (size/z), 'treemap'/'sunburst' for "
                            "hierarchical part-to-whole breakdowns (needs 'path'), 'waterfall' for "
                            "cumulative change, 'sankey' for flow between stages (needs a 'source', "
                            "'target', 'value' table via data_code), 'funnel' for stage drop-off, "
                            "'radar'/'polar_bar' for multi-metric comparison, 'gauge'/'bullet' for a "
                            "single KPI value, 'candlestick' for OHLC financial data (needs 'date', "
                            "'open', 'high', 'low', 'close' columns via data_code), 'choropleth' for "
                            "a country/region map (x = country names, y = value), 'density_heatmap'/"
                            "'contour' for 2D point density, 'strip'/'ecdf' for distribution detail. "
                            "IMPORTANT: if the chart needs data that isn't already a plain column in "
                            "the dataset (e.g. extracting year from a date column, counting "
                            "occurrences, a groupby aggregation, or building the source/target/value "
                            "table for a sankey), you MUST provide 'data_code': short pandas code "
                            "(same rules as run_pandas_code: operates on `df`, assigns a DataFrame to "
                            "`result`) that computes the exact table to chart. Then set x/y/color/"
                            "size/z to column names of that computed result, not the raw dataset. Do "
                            "not call run_pandas_code first and then build_chart separately for this "
                            "— compute and chart in one build_chart call via data_code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chart_type": {
                        "type": "string",
                        "enum": [
                            "line", "area",
                            "bar", "bar_horizontal", "stacked_bar", "grouped_bar", "polar_bar",
                            "scatter", "bubble", "scatter_3d", "density_heatmap", "contour",
                            "pie", "donut", "treemap", "sunburst",
                            "histogram", "box", "violin", "strip", "ecdf",
                            "heatmap",
                            "funnel", "waterfall", "sankey",
                            "radar", "gauge", "bullet",
                            "candlestick",
                            "choropleth",
                        ],
                    },
                    "x": {"type": "string", "description": "Column for x-axis / categories."},
                    "y": {"type": "string", "description": "Column for y-axis / values."},
                    "z": {"type": "string", "description": "Column for the z-axis (scatter_3d only)."},
                    "color": {"type": "string", "description": "Optional column to color/group by."},
                    "size": {"type": "string", "description": "Column for point size (bubble/scatter/scatter_3d only)."},
                    "path": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Ordered hierarchy columns for treemap/sunburst, e.g. ['region', 'product'].",
                    },
                    "data_code": {
                        "type": "string",
                        "description": "Optional pandas code operating on `df` that computes the "
                                        "table to chart, ending with `result = ...` (a DataFrame). "
                                        "Use this whenever the needed columns don't already exist "
                                        "as-is in the dataset.",
                    },
                    "title": {"type": "string", "description": "Chart title."},
                },
                "required": ["chart_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_anomalies",
            "description": "Run deterministic outlier detection (IQR-based) on a numeric column, "
                            "or a time trend with % change if a date column and value column are given.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string", "description": "Numeric column to check for outliers."},
                    "date_column": {"type": "string", "description": "Optional date column for trend analysis."},
                    "value_column": {"type": "string", "description": "Value column to trend, if date_column is given."},
                },
                "required": [],
            },
        },
    },
]


class ToolExecutionError(Exception):
    pass


def _df_preview(obj, max_rows: int = 8) -> str:
    if isinstance(obj, pd.DataFrame):
        if obj.empty:
            return "(empty result)"
        return obj.head(max_rows).to_string()
    if isinstance(obj, pd.Series):
        return obj.head(max_rows).to_string()
    return str(obj)


def execute_tool(name: str, arguments: dict, df: pd.DataFrame, sandbox_timeout: int = 5):
    """
    Dispatches a tool call to the right deterministic module.
    Returns (text_for_llm: str, raw_payload: Any-for-UI-rendering-like-a-chart-or-df)
    """
    try:
        if name == "get_profile":
            shape = profiling.basic_shape(df)
            cols = profiling.column_profile(df)
            text = f"Shape: {shape}\n\nColumn profile:\n{cols.to_string(index=False)}"
            return text, cols

        if name == "run_pandas_code":
            code = arguments.get("code", "")
            result = sandbox.run_pandas_code(code, df, timeout_seconds=sandbox_timeout)
            if isinstance(result, pd.DataFrame):
                result = df_utils.dedupe_columns(result)
            return f"Result:\n{_df_preview(result)}", result

        if name == "run_sql":
            sql = arguments.get("sql", "")
            result = sql_engine.run_sql(sql, df)
            result = df_utils.dedupe_columns(result)
            return f"Query result:\n{_df_preview(result)}", result



        if name == "build_chart":
            data_code = arguments.get("data_code")
            chart_source_df = df
            if data_code:
                computed = sandbox.run_pandas_code(data_code, df, timeout_seconds=sandbox_timeout)
                if isinstance(computed, pd.Series):
                    computed = computed.reset_index()
                if not isinstance(computed, pd.DataFrame):
                    return (
                        "build_chart failed: 'data_code' must produce a DataFrame "
                        "(or Series) in `result`, not a scalar.",
                        None,
                    )
                chart_source_df = df_utils.dedupe_columns(computed)

            fig = charts.build_chart(
                chart_source_df,
                chart_type=arguments.get("chart_type", ""),
                x=arguments.get("x"),
                y=arguments.get("y"),
                color=arguments.get("color"),
                size=arguments.get("size"),
                z=arguments.get("z"),
                path=arguments.get("path"),
                title=arguments.get("title"),
            )
            return "Chart created successfully and shown to the user.", fig

        if name == "detect_anomalies":
            column = arguments.get("column")
            date_col = arguments.get("date_column")
            value_col = arguments.get("value_column")
            if date_col and value_col:
                trend = anomaly.trend_over_time(df, date_col, value_col)
                summary = anomaly.pct_change_summary(trend)
                text = f"Trend summary: {json.dumps(summary)}\n\nPeriods:\n{trend.tail(12).to_string(index=False)}"
                return text, trend
            if column:
                outliers = anomaly.detect_outliers_iqr(df, column)
                text = f"Found {len(outliers)} outlier rows (IQR method) in '{column}'.\n{_df_preview(outliers)}"
                return text, outliers
            return "No column specified for anomaly detection.", None

        raise ToolExecutionError(f"Unknown tool: {name}")

    except Exception as e:  # noqa: BLE001 - convert to a friendly message for the LLM
        return f"Tool '{name}' failed with error: {e}", None
