"""
Deterministic cleaning-suggestion engine. Produces concrete, actionable
suggestions the user can choose to apply — no LLM guessing on what's "wrong".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CleaningSuggestion:
    id: str
    column: str | None
    issue: str
    detail: str
    action_label: str


def _iqr_outlier_count(s: pd.Series) -> int:
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0 or pd.isna(iqr):
        return 0
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(((s < lower) | (s > upper)).sum())


def generate_suggestions(df: pd.DataFrame) -> list[CleaningSuggestion]:
    suggestions: list[CleaningSuggestion] = []

    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        suggestions.append(
            CleaningSuggestion(
                id="drop_duplicates",
                column=None,
                issue="Duplicate rows",
                detail=f"{dup_count} exact duplicate rows found.",
                action_label="Drop duplicate rows",
            )
        )

    for col in df.columns:
        s = df[col]
        missing_pct = s.isna().mean() * 100
        if missing_pct > 0:
            suggestions.append(
                CleaningSuggestion(
                    id=f"missing_{col}",
                    column=col,
                    issue="Missing values",
                    detail=f"'{col}' has {missing_pct:.1f}% missing values.",
                    action_label=(
                        f"Drop rows missing '{col}'"
                        if missing_pct < 5
                        else f"Impute '{col}' (median/mode)"
                    ),
                )
            )

        if pd.api.types.is_numeric_dtype(s):
            n_out = _iqr_outlier_count(s.dropna())
            if n_out > 0:
                suggestions.append(
                    CleaningSuggestion(
                        id=f"outliers_{col}",
                        column=col,
                        issue="Potential outliers",
                        detail=f"'{col}' has {n_out} values outside 1.5×IQR range.",
                        action_label=f"Review/cap outliers in '{col}'",
                    )
                )

        if s.dtype == object:
            trimmed = s.dropna().astype(str).str.strip()
            if not trimmed.empty and (trimmed != s.dropna().astype(str)).any():
                suggestions.append(
                    CleaningSuggestion(
                        id=f"whitespace_{col}",
                        column=col,
                        issue="Whitespace inconsistency",
                        detail=f"'{col}' has values with leading/trailing whitespace.",
                        action_label=f"Trim whitespace in '{col}'",
                    )
                )
            # inconsistent casing for likely-categorical text
            if s.nunique(dropna=True) < max(50, len(s) * 0.5):
                lower_unique = trimmed.str.lower().nunique()
                if lower_unique < trimmed.nunique():
                    suggestions.append(
                        CleaningSuggestion(
                            id=f"casing_{col}",
                            column=col,
                            issue="Inconsistent text casing",
                            detail=(
                                f"'{col}' may have case-inconsistent categories "
                                f"(e.g. 'NY' vs 'ny')."
                            ),
                            action_label=f"Normalize casing in '{col}'",
                        )
                    )

    return suggestions


def apply_suggestion(df: pd.DataFrame, suggestion_id: str) -> pd.DataFrame:
    """Applies a single suggestion by id and returns a NEW dataframe (does not mutate)."""
    df = df.copy()

    if suggestion_id == "drop_duplicates":
        return df.drop_duplicates()

    if suggestion_id.startswith("missing_"):
        col = suggestion_id.removeprefix("missing_")
        missing_pct = df[col].isna().mean() * 100
        if missing_pct < 5:
            return df.dropna(subset=[col])
        if pd.api.types.is_numeric_dtype(df[col]):
            return df.assign(**{col: df[col].fillna(df[col].median())})
        mode = df[col].mode(dropna=True)
        fill_val = mode.iloc[0] if not mode.empty else "Unknown"
        return df.assign(**{col: df[col].fillna(fill_val)})

    if suggestion_id.startswith("outliers_"):
        col = suggestion_id.removeprefix("outliers_")
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        df[col] = df[col].clip(lower=lower, upper=upper)
        return df

    if suggestion_id.startswith("whitespace_"):
        col = suggestion_id.removeprefix("whitespace_")
        df[col] = df[col].astype(str).str.strip()
        return df

    if suggestion_id.startswith("casing_"):
        col = suggestion_id.removeprefix("casing_")
        df[col] = df[col].astype(str).str.strip().str.title()
        return df

    return df
