"""
Deterministic anomaly & trend detection. Pure statistics — no LLM.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def detect_outliers_zscore(df: pd.DataFrame, column: str, threshold: float = 3.0) -> pd.DataFrame:
    s = df[column].dropna()
    if s.empty or s.std(ddof=0) == 0:
        return df.iloc[0:0]
    z = (s - s.mean()) / s.std(ddof=0)
    outlier_idx = z[abs(z) > threshold].index
    return df.loc[outlier_idx]


def detect_outliers_iqr(df: pd.DataFrame, column: str) -> pd.DataFrame:
    s = df[column].dropna()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return df.iloc[0:0]
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (df[column] < lower) | (df[column] > upper)
    return df[mask]


def trend_over_time(df: pd.DataFrame, date_col: str, value_col: str, freq: str = "M") -> pd.DataFrame:
    """Aggregates `value_col` by time period for trend analysis. Returns real,
    computed aggregates (sum + mean + count) per period."""
    work = df[[date_col, value_col]].dropna().copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=[date_col])
    grouped = (
        work.set_index(date_col)[value_col]
        .resample(freq)
        .agg(["sum", "mean", "count"])
        .reset_index()
    )
    return grouped


def pct_change_summary(trend_df: pd.DataFrame, value_col: str = "sum") -> dict:
    """Simple period-over-period % change summary based on real aggregated values."""
    if trend_df.empty or len(trend_df) < 2:
        return {"available": False}
    first, last = trend_df[value_col].iloc[0], trend_df[value_col].iloc[-1]
    if first == 0 or pd.isna(first) or pd.isna(last):
        return {"available": False}
    pct = ((last - first) / abs(first)) * 100
    return {
        "available": True,
        "first_period_value": round(float(first), 2),
        "last_period_value": round(float(last), 2),
        "pct_change": round(float(pct), 2),
    }
