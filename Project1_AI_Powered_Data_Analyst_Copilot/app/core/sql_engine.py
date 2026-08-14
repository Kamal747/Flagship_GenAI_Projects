"""
SQL execution engine backed by DuckDB, running directly against the real
in-memory dataframe (no data duplication into an external DB). This lets the
copilot answer "generate and run SQL" requests with real, verifiable results.
"""
from __future__ import annotations

import duckdb
import pandas as pd


class SQLExecutionError(Exception):
    pass


FORBIDDEN_KEYWORDS = (
    "attach", "copy", "install", "load", "pragma", "call ", "export",
    "create table", "create or replace", "insert into", "update ", "delete from",
    "drop table", "drop view",
)


def _validate_sql(sql: str) -> None:
    lowered = sql.strip().lower()
    if not lowered.startswith("select") and not lowered.startswith("with"):
        raise SQLExecutionError("Only SELECT / WITH (read-only) queries are allowed.")
    for kw in FORBIDDEN_KEYWORDS:
        if kw in lowered:
            raise SQLExecutionError(f"Query contains a disallowed operation: '{kw.strip()}'.")


def run_sql(sql: str, df: pd.DataFrame, table_name: str = "dataset") -> pd.DataFrame:
    """
    Executes a read-only SQL query against `df`, exposed as a table called
    `table_name` (default 'dataset'). Returns the result as a DataFrame.
    """
    _validate_sql(sql)
    con = duckdb.connect(database=":memory:")
    try:
        con.register(table_name, df)
        try:
            result = con.execute(sql).fetchdf()
        except Exception as e:  # noqa: BLE001
            raise SQLExecutionError(f"SQL execution failed: {e}")
        return result
    finally:
        con.close()

