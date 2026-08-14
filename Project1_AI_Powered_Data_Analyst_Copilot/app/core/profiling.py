"""
Deterministic dataset profiling. No LLM calls here — pure Pandas/NumPy,
so every number is 100% traceable to the real data.
"""
from __future__ import annotations

import pandas as pd


def basic_shape(df: pd.DataFrame) -> dict:
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
        "duplicate_rows": int(df.duplicated().sum()),
    }


def column_profile(df: pd.DataFrame) -> pd.DataFrame:
    """One row per column: dtype, missing %, unique count, sample stats."""
    rows = []
    n = len(df)
    for col in df.columns:
        s = df[col]
        missing = int(s.isna().sum())
        missing_pct = round((missing / n) * 100, 2) if n else 0.0
        dtype = str(s.dtype)
        unique = int(s.nunique(dropna=True))

        entry = {
            "column": col,
            "dtype": dtype,
            "missing": missing,
            "missing_pct": missing_pct,
            "unique": unique,
        }

        if pd.api.types.is_numeric_dtype(s):
            desc = s.describe()
            entry.update(
                {
                    "mean": round(float(desc.get("mean", float("nan"))), 4)
                    if n else None,
                    "min": desc.get("min"),
                    "max": desc.get("max"),
                    "std": round(float(desc.get("std", float("nan"))), 4) if n else None,
                }
            )
        elif pd.api.types.is_datetime64_any_dtype(s):
            entry.update({"min": s.min(), "max": s.max()})
        else:
            top = s.mode(dropna=True)
            entry.update({"top_value": top.iloc[0] if not top.empty else None})

        rows.append(entry)
    return pd.DataFrame(rows)


def numeric_correlations(df: pd.DataFrame) -> pd.DataFrame | None:
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        return None
    return numeric_df.corr(numeric_only=True).round(3)


def full_profile_summary(df: pd.DataFrame) -> str:
    """A compact, LLM-friendly text summary of the dataset — used as grounding
    context so the model reasons about real structure instead of guessing."""
    shape = basic_shape(df)
    cols = column_profile(df)
    lines = [
        f"Rows: {shape['rows']}, Columns: {shape['columns']}, "
        f"Duplicate rows: {shape['duplicate_rows']}, Memory: {shape['memory_mb']} MB",
        "Columns:",
    ]
    for _, r in cols.iterrows():
        lines.append(
            f"- {r['column']} ({r['dtype']}): missing={r['missing']} "
            f"({r['missing_pct']}%), unique={r['unique']}"
        )
    return "\n".join(lines)
