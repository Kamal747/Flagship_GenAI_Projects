"""
Restricted sandbox for executing LLM-generated Pandas code.

This is the core trust mechanism of the app: the LLM never states a number
itself. It writes a small Pandas expression/snippet, this module executes it
against the REAL dataframe under tight restrictions, and only the actual
computed result is returned. The LLM only narrates that result afterward.

Restrictions:
- No imports, no dunder access, no file/network/system access.
- Only a small whitelist of builtins and pandas/numpy are available.
- Code must assign its final answer to a variable named `result`.
- Wall-clock timeout to prevent runaway/expensive code.
"""
from __future__ import annotations

import ast
import multiprocessing as mp
import queue

import numpy as np
import pandas as pd

FORBIDDEN_NODE_TYPES = (ast.Import, ast.ImportFrom)
FORBIDDEN_NAME_PARTS = ("__", "os", "sys", "subprocess", "open", "eval", "exec", "compile", "input")

SAFE_BUILTINS = {
    "len": len, "range": range, "sum": sum, "min": min, "max": max,
    "sorted": sorted, "round": round, "abs": abs, "list": list, "dict": dict,
    "set": set, "tuple": tuple, "str": str, "int": int, "float": float,
    "bool": bool, "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
    "True": True, "False": False, "None": None,
}


class SandboxError(Exception):
    pass


def _static_safety_check(code: str) -> None:
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as e:
        raise SandboxError(f"Generated code has a syntax error: {e}")

    for node in ast.walk(tree):
        if isinstance(node, FORBIDDEN_NODE_TYPES):
            raise SandboxError("Imports are not allowed in generated code.")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise SandboxError("Access to dunder attributes is not allowed.")
        if isinstance(node, ast.Name) and any(f in node.id for f in FORBIDDEN_NAME_PARTS):
            raise SandboxError(f"Use of restricted name '{node.id}' is not allowed.")


def _worker(code: str, df: pd.DataFrame, out_q: "mp.Queue") -> None:
    try:
        local_scope = {"df": df.copy(), "pd": pd, "np": np}
        global_scope = {"__builtins__": SAFE_BUILTINS}
        exec(code, global_scope, local_scope)  # noqa: S102 - sandboxed on purpose
        result = local_scope.get("result", None)
        # Normalize common pandas objects into JSON/repr-friendly forms
        if isinstance(result, (pd.DataFrame, pd.Series)):
            out_q.put(("ok", result))
        elif isinstance(result, (np.integer, np.floating)):
            out_q.put(("ok", result.item()))
        else:
            out_q.put(("ok", result))
    except Exception as e:  # noqa: BLE001
        out_q.put(("error", str(e)))


def run_pandas_code(code: str, df: pd.DataFrame, timeout_seconds: int = 5):
    """
    Executes LLM-generated pandas code against `df` in a subprocess with a
    timeout, after a static AST safety check. Code MUST set a `result` var.
    Returns the value of `result` (DataFrame, Series, scalar, etc).
    Raises SandboxError on any violation, timeout, or runtime error.
    """
    if "result" not in code:
        raise SandboxError("Generated code must assign its answer to a variable named `result`.")

    _static_safety_check(code)

    ctx = mp.get_context("spawn")
    out_q: mp.Queue = ctx.Queue()
    proc = ctx.Process(target=_worker, args=(code, df, out_q))
    proc.start()
    proc.join(timeout=timeout_seconds)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        raise SandboxError(f"Code execution timed out after {timeout_seconds}s.")

    try:
        status, payload = out_q.get_nowait()
    except queue.Empty:
        raise SandboxError("Code execution failed with no result (possible crash).")

    if status == "error":
        raise SandboxError(f"Error while running the generated code: {payload}")
    return payload

