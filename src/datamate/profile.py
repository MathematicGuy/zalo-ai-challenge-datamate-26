"""Cheap structural profiling of raw tables, used by the EDA notebook."""
from __future__ import annotations

import re

import pandas as pd

from .io import RawTable, Readme
from .text import fold

# Patterns a string cell can take. Order matters only for display.
CELL_PATTERNS = {
    "int": r"^-?\d+$",
    "decimal": r"^-?\d+\.\d+$",
    "thousands_dot": r"^-?\d{1,3}(\.\d{3})+$",
    "thousands_comma": r"^-?\d{1,3}(,\d{3})+$",
    "money_dong": r"^-?[\d.,]+\s?(đ|đồng|vnđ|vnd)$",
    "money_k": r"^-?\d+([.,]\d+)?\s?k$",
    "money_tr": r"^-?\d+([.,]\d+)?\s?tr(\d+)?$",
    "money_word": r"^-?\d+([.,]\d+)?\s?(triệu|nghìn|ngàn|tỷ)$",
    "num_with_unit": r"^\d+([.,]\d+)?\s+\D+$",
    "date_dmy": r"^\d{1,2}/\d{1,2}/\d{4}$",
    "date_dmy_dash": r"^\d{1,2}-\d{1,2}-\d{4}$",
    "date_dm": r"^\d{1,2}/\d{1,2}$",
    "date_iso": r"^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2})?)?$",
    "date_thg": r"^\d{1,2} thg \d{1,2}( \d{4})?$",
    "time_hm": r"^\d{1,2}:\d{2}(:\d{2})?$",
    "bool_word": r"^(true|false|có|không|x)$",
}
_COMPILED = {k: re.compile(v, re.I) for k, v in CELL_PATTERNS.items()}


def cell_kind(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "empty"
    if isinstance(v, pd.Timestamp) or hasattr(v, "year") and hasattr(v, "month"):
        return "datetime_obj"
    if isinstance(v, bool):
        return "bool_obj"
    if isinstance(v, (int, float)):
        return "number_obj"
    s = str(v).strip()
    if not s:
        return "empty"
    for k, rx in _COMPILED.items():
        if rx.match(s):
            return k
    return "text"


def readme_columns_for(readme: Readme, table: RawTable) -> dict[str, str]:
    """README column map for a raw table: matches `file`, `file, sheet X`, `pattern<...>.xlsx` or sqlite `table`."""
    names = [table.file, table.sheet or ""]
    best = {}
    for key, cols in readme.columns.items():
        fpart, _, spart = key.partition(", sheet ")
        rx = "^" + re.sub(r"\\<[^>]*\\>", ".+", re.escape(fpart.strip())) + "$"
        if (re.match(rx, table.file) or (table.sheet and fpart.strip() == table.sheet)) and \
                (not spart or spart.strip() == table.sheet):
            best = cols
        elif not best and fpart.strip() in names:
            best = cols
    return best


def find_header_row(df: pd.DataFrame, expected: list[str], scan: int = 10) -> tuple[int, float]:
    """Row (within the first `scan`) whose cells best cover the README column names; returns (row, coverage)."""
    if not expected:
        return 0, float("nan")
    want = {fold(c) for c in expected}
    best = (0, -1.0)
    for i in range(min(scan, len(df))):
        got = {fold(v) for v in df.iloc[i].tolist() if isinstance(v, str)}
        cov = len(want & got) / len(want)
        if cov > best[1]:
            best = (i, cov)
    return best


def with_header(df: pd.DataFrame, row: int) -> pd.DataFrame:
    cols = [str(c) if c is not None and not (isinstance(c, float) and pd.isna(c)) else f"_c{i}"
            for i, c in enumerate(df.iloc[row].tolist())]
    return df.iloc[row + 1:].set_axis(cols, axis=1).reset_index(drop=True)


def role_columns(readme_cols: dict[str, str], role_prefix: str) -> list[str]:
    return [c for c, r in readme_cols.items() if r.startswith(role_prefix)]
