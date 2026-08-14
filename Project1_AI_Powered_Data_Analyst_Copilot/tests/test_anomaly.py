import pandas as pd

from app.core import anomaly


def test_iqr_outlier_detection_finds_known_outlier():
    df = pd.DataFrame({"value": [10, 11, 12, 13, 12, 11, 500]})
    outliers = anomaly.detect_outliers_iqr(df, "value")
    assert 500 in outliers["value"].values


def test_iqr_no_outliers_when_uniform():
    df = pd.DataFrame({"value": [10, 10, 10, 10]})
    outliers = anomaly.detect_outliers_iqr(df, "value")
    assert outliers.empty


def test_pct_change_summary_computes_correctly():
    trend_df = pd.DataFrame({"sum": [100, 150]})
    summary = anomaly.pct_change_summary(trend_df)
    assert summary["available"] is True
    assert summary["pct_change"] == 50.0
