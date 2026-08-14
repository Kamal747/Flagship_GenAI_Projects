import pandas as pd
import pytest

from app.core import sandbox


@pytest.fixture
def sample_df():
    return pd.DataFrame({"a": [1, 2, 3, 4], "b": ["x", "y", "x", "y"]})


def test_valid_code_runs(sample_df):
    result = sandbox.run_pandas_code("result = df['a'].sum()", sample_df)
    assert result == 10


def test_groupby_returns_series(sample_df):
    result = sandbox.run_pandas_code("result = df.groupby('b')['a'].sum()", sample_df)
    assert isinstance(result, pd.Series)
    assert result["x"] == 4


def test_missing_result_var_rejected(sample_df):
    with pytest.raises(sandbox.SandboxError):
        sandbox.run_pandas_code("x = df['a'].sum()", sample_df)


def test_import_rejected(sample_df):
    with pytest.raises(sandbox.SandboxError):
        sandbox.run_pandas_code("import os\nresult = 1", sample_df)


def test_dunder_access_rejected(sample_df):
    with pytest.raises(sandbox.SandboxError):
        sandbox.run_pandas_code("result = df.__class__", sample_df)


def test_os_reference_rejected(sample_df):
    with pytest.raises(sandbox.SandboxError):
        sandbox.run_pandas_code("result = os.getcwd()", sample_df)


def test_syntax_error_handled(sample_df):
    with pytest.raises(sandbox.SandboxError):
        sandbox.run_pandas_code("result = df[[", sample_df)
