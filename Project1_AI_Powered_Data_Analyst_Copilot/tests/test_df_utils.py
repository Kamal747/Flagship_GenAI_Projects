import pandas as pd
import pytest

from app.core import df_utils, tools


def test_dedupe_columns_renames_duplicates():
    df = pd.DataFrame([[1, 2]], columns=["count", "count"])
    result = df_utils.dedupe_columns(df)
    assert list(result.columns) == ["count", "count_1"]


def test_dedupe_columns_handles_triple_duplicate():
    df = pd.DataFrame([[1, 2, 3]], columns=["x", "x", "x"])
    result = df_utils.dedupe_columns(df)
    assert list(result.columns) == ["x", "x_1", "x_2"]


def test_dedupe_columns_noop_when_unique():
    df = pd.DataFrame({"a": [1], "b": [2]})
    result = df_utils.dedupe_columns(df)
    assert list(result.columns) == ["a", "b"]
    assert result is df  # unchanged input returned as-is, no unnecessary copy


def test_dedupe_columns_passthrough_non_dataframe():
    assert df_utils.dedupe_columns(42) == 42
    assert df_utils.dedupe_columns(None) is None


def test_run_pandas_code_with_duplicate_result_columns_is_sanitized():
    # Reproduces the reported bug: combining two aggregations that both end
    # up named 'count' (e.g. via pd.concat) yields duplicate column labels.
    df = pd.DataFrame({"year": [2020, 2020, 2021, 2021, 2021]})
    code = (
        "a = df.groupby('year').size().reset_index(name='count')\n"
        "b = df.groupby('year').size().reset_index(name='count')\n"
        "result = pd.concat([a.set_index('year'), b.set_index('year')], axis=1).reset_index()"
    )
    text, payload = tools.execute_tool("run_pandas_code", {"code": code}, df)
    assert isinstance(payload, pd.DataFrame)
    assert not payload.columns.duplicated().any()
    assert list(payload.columns) == ["year", "count", "count_1"]


def test_build_chart_data_code_with_duplicate_columns_is_sanitized():
    df = pd.DataFrame({"year": [2020, 2020, 2021, 2021, 2021]})
    data_code = (
        "a = df.groupby('year').size().reset_index(name='count')\n"
        "b = df.groupby('year').size().reset_index(name='count')\n"
        "result = pd.concat([a.set_index('year'), b.set_index('year')], axis=1).reset_index()"
    )
    args = {
        "chart_type": "bar",
        "data_code": data_code,
        "x": "year",
        "y": "count",
    }
    text, fig = tools.execute_tool("build_chart", args, df)
    assert "successfully" in text
    assert fig is not None
