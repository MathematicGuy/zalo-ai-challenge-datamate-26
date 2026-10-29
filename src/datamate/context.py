"""Compact text description of a business's tables, for the prompt.

Per column it shows what a careful analyst would look up before writing code: the README's meaning
of the column, which helper reads its cells, the date range it covers, and, for low-cardinality
columns, every distinct value (statuses, channels, payment methods are spelled many ways).
"""
from __future__ import annotations

import datetime
import math
from types import SimpleNamespace

import pandas as pd

from .helpers import fold, to_date, to_number, year_of
from .profile import readme_columns_for
from .tables import Business

MAX_DISTINCT_LISTED = 20
SAMPLE = 3000


def _fmt(v) -> str:
    if isinstance(v, datetime.datetime):  # includes pd.Timestamp; shown as datetime(...) to keep it apart from str
        return f"datetime({v.isoformat(sep=' ')})"
    s = repr(v)
    return s if len(s) <= 40 else s[:37] + "...'"


def _parse_profile(values: list, col: pd.Series, year: int | None) -> str:
    """Which helper reads the string cells of a column (judged on `values`, a sample), and the date
    range over the whole column if they are dates."""
    strs = [v for v in values if isinstance(v, str)]
    if not strs:
        return ""
    words = sum(1 for s in strs if not any(ch.isdigit() for ch in s))
    if words / len(strs) > 0.5:
        return ""
    num_ok = sum(not math.isnan(to_number(s)) for s in strs) / len(strs)
    date_ok = sum(not pd.isna(to_date(v, year)) for v in values) / len(values)
    if date_ok >= 0.5 and date_ok >= num_ok - 0.2:
        note = " with year from the sheet name" if year and any(fold(s).count("/") == 1 for s in strs[:200]) else ""
        return f"to_date{note} reads {date_ok:.0%}, {_date_range(col, year)}"
    if num_ok >= 0.5:
        return f"to_number reads {num_ok:.0%}"
    return ""


def _date_range(col: pd.Series, year: int | None = None) -> str:
    """min..max over every distinct value of the column (not a sample: a head sample hides the span)."""
    ds = [d for d in (to_date(v, year) for v in col.dropna().unique()) if not pd.isna(d)]
    return f"range {min(ds).date()}..{max(ds).date()}" if ds else ""


def describe_frame(name: str, df: pd.DataFrame, roles: dict[str, str], n_examples: int = 4) -> str:
    year = year_of(name)
    lines = [f"### tables[{name!r}]  ({len(df)} rows x {df.shape[1]} cols)"]
    for c in df.columns:
        col = df[c]
        nonnull = col.dropna()
        sample = nonnull.head(SAMPLE).tolist()
        kinds = sorted({type(v).__name__ for v in sample})
        nunique = nonnull.nunique()
        blank = f", {col.isna().mean():.0%} blank" if col.isna().any() else ""
        role = f" — README: {roles[str(c)]}" if str(c) in roles else ""
        head = f"- {c!r}{role} [{'/'.join(kinds) or 'empty'}{blank}, {nunique} distinct]"
        if 0 < nunique <= MAX_DISTINCT_LISTED and any(isinstance(v, str) for v in sample):
            vc = nonnull.astype(str).value_counts()
            lines.append(head + ": ALL values " + ", ".join(f"{_fmt(k)}×{n}" for k, n in vc.items()))
            continue
        ex = list(dict.fromkeys(sample))[:n_examples]
        prof = _parse_profile(sample, nonnull, year) if "str" in kinds else (
            _date_range(nonnull) if any(k in kinds for k in ("datetime", "Timestamp")) else "")
        lines.append(head + (f" ({prof})" if prof else "") + ": " + ", ".join(_fmt(v) for v in ex))
    return "\n".join(lines)


def _roles_for(biz: Business, name: str) -> dict[str, str]:
    file, _, sheet = name.partition("::")
    t = SimpleNamespace(file=file, sheet=sheet or None)
    return readme_columns_for(biz.readme, t)


def describe_business(biz: Business) -> str:
    parts = [f"## README.md of {biz.biz_id}\n", biz.readme.text.strip(), "\n## Loaded tables",
             "Keys of `tables` (use exactly these): " + ", ".join(repr(n) for n in biz.frames)]
    groups: dict[tuple, list[str]] = {}
    for n, df in biz.frames.items():
        groups.setdefault(tuple(map(str, df.columns)), []).append(n)
    for names in groups.values():
        parts.append(describe_frame(names[0], biz.frames[names[0]], _roles_for(biz, names[0])))
        if len(names) > 1:
            rows = sum(len(biz.frames[n]) for n in names)
            parts.append(f"  (+{len(names) - 1} more tables with the same columns, {rows} rows in total: "
                         + ", ".join(repr(n) for n in names[1:]) + ")")
            parts += _group_date_ranges(biz, names)
    return "\n".join(parts)


def _group_date_ranges(biz: Business, names: list[str], per_table: int = 300) -> list[str]:
    """Date range of each date column across every table of a same-columns group (head and tail sampled)."""
    out = []
    for c in biz.frames[names[0]].columns:
        lo = hi = None
        for n in names:
            col = biz.frames[n][c].dropna()
            vals = pd.concat([col.head(per_table), col.tail(per_table)]).tolist()
            ds = [d for d in (to_date(v, year_of(n)) for v in vals) if not pd.isna(d)]
            if len(ds) < 0.5 * max(len(vals), 1):
                lo = None
                break
            lo, hi = min([lo or ds[0], *ds]), max([hi or ds[0], *ds])
        if lo is not None:
            out.append(f"  date range of {c!r} across all {len(names)} tables: {lo.date()}..{hi.date()}")
    return out
