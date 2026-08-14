"""
Handles file upload, parsing, and initial dtype inference for CSV/Excel files.
Pure, deterministic — no LLM involved.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import pandas as pd


class DataLoadError(Exception):
    """Raised when a file cannot be parsed into a usable dataframe."""


@dataclass
class LoadedDataset:
    name: str
    df: pd.DataFrame
    sheet_names: list[str] = field(default_factory=list)
    original_shape: tuple[int, int] = (0, 0)


def _infer_and_clean_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Light, safe dtype inference: strips whitespace from headers,
    attempts numeric/date coercion only where it doesn't lose data."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    for col in df.columns:
        if df[col].dtype == object:
            stripped = df[col].astype(str).str.strip()
            # Try numeric coercion
            numeric = pd.to_numeric(stripped, errors="coerce")
            non_null_original = stripped.notna().sum()
            if non_null_original > 0 and numeric.notna().sum() / max(non_null_original, 1) > 0.9:
                df[col] = numeric
                continue
            # Try datetime coercion
            try:
                dt = pd.to_datetime(stripped, errors="coerce", format=None)
                if non_null_original > 0 and dt.notna().sum() / max(non_null_original, 1) > 0.9:
                    df[col] = dt
                    continue
            except Exception:
                pass
            df[col] = stripped
    return df


def load_file(uploaded_file, max_upload_mb: int = 200) -> dict[str, LoadedDataset]:
    """
    Loads an uploaded CSV or Excel file.
    Returns a dict of sheet_name -> LoadedDataset (CSV = single entry named 'Sheet1').
    Raises DataLoadError on failure.
    """
    size_mb = getattr(uploaded_file, "size", 0) / (1024 * 1024)
    if size_mb > max_upload_mb:
        raise DataLoadError(
            f"File is {size_mb:.1f} MB, which exceeds the {max_upload_mb} MB limit."
        )

    filename = uploaded_file.name
    raw_bytes = uploaded_file.read()

    try:
        if filename.lower().endswith(".csv"):
            df = pd.read_csv(io.BytesIO(raw_bytes))
            df = _infer_and_clean_dtypes(df)
            return {
                "Sheet1": LoadedDataset(
                    name=filename, df=df, sheet_names=["Sheet1"], original_shape=df.shape
                )
            }
        elif filename.lower().endswith((".xlsx", ".xls")):
            xls = pd.ExcelFile(io.BytesIO(raw_bytes))
            result = {}
            for sheet in xls.sheet_names:
                df = xls.parse(sheet)
                if df.empty:
                    continue
                df = _infer_and_clean_dtypes(df)
                result[sheet] = LoadedDataset(
                    name=filename, df=df, sheet_names=xls.sheet_names, original_shape=df.shape
                )
            if not result:
                raise DataLoadError("No non-empty sheets found in the Excel file.")
            return result
        else:
            raise DataLoadError(
                f"Unsupported file type: '{filename}'. Please upload a .csv, .xlsx, or .xls file."
            )
    except DataLoadError:
        raise
    except pd.errors.EmptyDataError:
        raise DataLoadError("The file appears to be empty.")
    except pd.errors.ParserError as e:
        raise DataLoadError(f"Could not parse the file — it may be malformed. Details: {e}")
    except Exception as e:  # noqa: BLE001 - surface as friendly error
        raise DataLoadError(f"Unexpected error while reading the file: {e}")
