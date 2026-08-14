"""
Small, focused DataFrame safety utilities shared across tool outputs.
Kept separate from profiling/cleaning since this is about making ANY
dataframe (including ones from LLM-generated code) safe to render/return,
not about analysis logic.
"""
from __future__ import annotations

import pandas as pd


def dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renames duplicate column labels so the DataFrame has unique column names.
    This is needed because LLM-generated Pandas code can produce duplicate
    column names (e.g. calling `.reset_index(name='count')` when a `count`
    column already exists), which PyArrow/Streamlit's dataframe renderer
    rejects with a ValueError. First occurrence keeps its original name;
    subsequent duplicates get a `_1`, `_2`, ... suffix.
    """
    if not isinstance(df, pd.DataFrame) or not df.columns.duplicated().any():
        return df

    df = df.copy()
    seen: dict[str, int] = {}
    new_columns = []
    for col in df.columns:
        if col not in seen:
            seen[col] = 0
            new_columns.append(col)
        else:
            seen[col] += 1
            new_columns.append(f"{col}_{seen[col]}")
    df.columns = new_columns
    return df

