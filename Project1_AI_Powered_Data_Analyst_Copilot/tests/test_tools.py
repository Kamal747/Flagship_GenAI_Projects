import pandas as pd

from app.core import tools


def test_build_chart_with_data_code_derives_year():
    df = pd.DataFrame({
        "title": [f"Show {i}" for i in range(6)],
        "date_added": pd.to_datetime(
            ["2019-01-05", "2019-06-01", "2020-02-14", "2020-07-19", "2021-01-01", "2021-05-05"]
        ),
    })
    args = {
        "chart_type": "bar",
        "data_code": (
            "result = df.assign(year=df['date_added'].dt.year)"
            ".groupby('year').size().reset_index(name='count')"
        ),
        "x": "year",
        "y": "count",
    }
    text, fig = tools.execute_tool("build_chart", args, df)
    assert "successfully" in text
    assert fig is not None


def test_build_chart_data_code_rejects_non_dataframe_result():
    df = pd.DataFrame({"a": [1, 2, 3]})
    args = {"chart_type": "bar", "data_code": "result = df['a'].sum()", "x": "a", "y": "a"}
    text, fig = tools.execute_tool("build_chart", args, df)
    assert "must produce a DataFrame" in text
    assert fig is None


def test_build_chart_without_data_code_still_works():
    df = pd.DataFrame({"region": ["N", "S"], "revenue": [100, 200]})
    args = {"chart_type": "bar", "x": "region", "y": "revenue"}
    text, fig = tools.execute_tool("build_chart", args, df)
    assert "successfully" in text
    assert fig is not None

