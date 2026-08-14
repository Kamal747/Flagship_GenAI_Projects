import numpy as np
import pandas as pd

from app.core import profiling


def make_df():
    return pd.DataFrame(
        {
            "num": [1, 2, 3, np.nan, 5],
            "cat": ["a", "b", "a", "a", None],
        }
    )


def test_basic_shape():
    df = make_df()
    shape = profiling.basic_shape(df)
    assert shape["rows"] == 5
    assert shape["columns"] == 2
    assert shape["duplicate_rows"] == 0


def test_column_profile_missing_counts():
    df = make_df()
    prof = profiling.column_profile(df)
    num_row = prof[prof["column"] == "num"].iloc[0]
    assert num_row["missing"] == 1
    cat_row = prof[prof["column"] == "cat"].iloc[0]
    assert cat_row["missing"] == 1


def test_column_profile_numeric_mean_matches_pandas():
    df = make_df()
    prof = profiling.column_profile(df)
    num_row = prof[prof["column"] == "num"].iloc[0]
    assert round(num_row["mean"], 4) == round(df["num"].mean(), 4)


def test_numeric_correlations_none_when_insufficient_columns():
    df = pd.DataFrame({"only_one": [1, 2, 3]})
    assert profiling.numeric_correlations(df) is None

