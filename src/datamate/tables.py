"""Turn a business folder into named DataFrames with the header row applied.

No value parsing here: cells keep their raw form ('55k', '1/4', '2,825tr'). Baseline v0 leaves
parsing to the generated code; later versions add helpers.
"""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .io import Readme, RawTable, parse_readme, read_raw_tables
from .profile import find_header_row, readme_columns_for, with_header

CACHE = Path(os.environ.get("DATAMATE_CACHE", Path(__file__).resolve().parents[2] / "data" / "cache"))


@dataclass
class Business:
    biz_id: str
    readme: Readme
    frames: dict[str, pd.DataFrame]   # "file" | "file::sheet" | sqlite table name -> DataFrame
    header_rows: dict[str, int]


def _guess_header_row(df: pd.DataFrame, scan: int = 8) -> int:
    """First row that is mostly non-empty strings and is followed by data (skips group-header rows)."""
    ncols = df.shape[1]
    for i in range(min(scan, len(df) - 1)):
        vals = [v for v in df.iloc[i].tolist() if v is not None and not (isinstance(v, float) and pd.isna(v))]
        if len(vals) >= max(1, 0.6 * ncols) and all(isinstance(v, str) for v in vals):
            return i
    return 0


def _name(t: RawTable, multi_sheet: bool) -> str:
    if t.file.endswith(".sqlite"):
        return t.sheet
    return f"{t.file}::{t.sheet}" if multi_sheet and t.sheet else t.file


def load_business(biz_dir: Path) -> Business:
    readme = parse_readme(biz_dir / "README.md")
    raw = read_raw_tables(biz_dir)
    sheets_per_file = pd.Series([t.file for t in raw]).value_counts()
    frames, header_rows = {}, {}
    for t in raw:
        if t.file.endswith(".sqlite"):
            row = 0  # read_raw_tables puts the real column names on row 0
        else:
            cols = readme_columns_for(readme, t)
            row, cov = find_header_row(t.df, list(cols))
            if not cols or not cov:
                row = _guess_header_row(t.df)
        df = with_header(t.df, row).dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
        name = _name(t, sheets_per_file[t.file] > 1)
        frames[name], header_rows[name] = df, row
    return Business(biz_dir.name, readme, frames, header_rows)


def cache_path(split_root: Path, biz_id: str) -> Path:
    """Cache key includes the split folder name so train/public/private never collide."""
    return CACHE / f"biz_{split_root.name}_{biz_id}.pkl"


def load_business_cached(split_root: Path, biz_id: str) -> Business:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = cache_path(split_root, biz_id)
    if path.exists():
        return pickle.loads(path.read_bytes())
    biz = load_business(split_root / "data" / biz_id)
    path.write_bytes(pickle.dumps(biz))
    return biz
