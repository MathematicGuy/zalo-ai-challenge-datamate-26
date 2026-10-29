"""Rule-agnostic helpers injected into the generated code's namespace.

Only mechanics live here: reading Vietnamese money / quantity / date spellings, text folding,
forward-filling order lines, lunar-date lookup. Business rules (what counts as revenue, which
period "tháng này" means, ...) stay in the split's rulebook and are applied by the model, so a new
rulebook or a new industry needs no code change.
"""
from __future__ import annotations

import datetime as _dt
import math
import re
import unicodedata

import pandas as pd

# ---------------------------------------------------------------- text


def fold(s) -> str:
    """Lowercase, no accents (đ->d), single spaces: a matching key for text typed with or without
    diacritics. Never use it to produce an answer. None / NaN fold to ''."""
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return ""
    s = unicodedata.normalize("NFD", str(s).replace("đ", "d").replace("Đ", "D"))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn").lower()
    return re.sub(r"\s+", " ", s).strip()


def same_text(a, b) -> bool:
    """True when two names differ only in case, accents, spacing or punctuation."""
    k = lambda x: re.sub(r"[^0-9a-z]+", "", fold(x))
    return k(a) == k(b)


# ---------------------------------------------------------------- numbers

_WORD_MULT = {"k": 1e3, "nghin": 1e3, "ngan": 1e3, "tr": 1e6, "trieu": 1e6, "ty": 1e9, "ti": 1e9}
_CURRENCY = re.compile(r"(vnd|vnđ|đồng|dong|đ|d)\.?$")


def _plain_number(s: str) -> float | None:
    """'55.000' / '35,000' / '1.234.567' are thousands; '18.6' / '7,5' are decimals."""
    s = s.strip()
    if re.fullmatch(r"-?\d{1,3}(\.\d{3})+", s) or re.fullmatch(r"-?\d{1,3}(,\d{3})+", s):
        return float(re.sub(r"[.,]", "", s))
    if re.fullmatch(r"-?\d+([.,]\d+)?", s):
        return float(s.replace(",", "."))
    return None


def to_number(x, default=float("nan")):
    """Money or numeric cell -> float. Handles 55000, '55.000đ', '35,000', '55k', '7,5k', '2,825tr',
    '1tr312' (=1,312,000), '409tr663', '1,2 triệu', '2 tỷ', '-12.000'. Returns `default` if unreadable.
    A leading number followed by a unit word ('1 bình') also parses; use to_qty for quantities."""
    if x is None or isinstance(x, bool):
        return default
    if isinstance(x, (int, float)):
        return float(x) if not (isinstance(x, float) and math.isnan(x)) else default
    s = fold(x).replace(" ", "")
    if not s:
        return default
    s = _CURRENCY.sub("", s) if not re.search(r"\d(tr|k)\d*$", s) else s
    neg = s.startswith("-")
    s = s.lstrip("-+")
    # 1tr312 -> 1.312 tr ; 409tr663 -> 409.663 tr ; 2tr5 -> 2.5 tr
    m = re.fullmatch(r"(\d+)(tr|trieu|ty|ti|k)(\d+)", s)
    if m:
        v = float(f"{m[1]}.{m[3]}") * _WORD_MULT[m[2]]
        return -v if neg else v
    m = re.fullmatch(r"([\d.,]+)(k|nghin|ngan|tr|trieu|ty|ti)", s)
    if m:
        base = m[1]
        # '2,825tr' / '332,5972tr' / '1.2tr': the separator is a decimal point here
        v = float(base.replace(",", ".")) if base.count(",") + base.count(".") <= 1 else _plain_number(base)
        if v is None:
            return default
        v *= _WORD_MULT[m[2]]
        return -v if neg else v
    v = _plain_number(s)
    if v is None:
        m = re.match(r"([\d.,]+)\D", s)  # '1binh', '10hop'
        v = _plain_number(m[1]) if m else None
    if v is None:
        return default
    return -v if neg else v


def to_qty(x, default=float("nan")):
    """Quantity cell -> number: 2, '2', '1 bình', '2,5 kg', '10 hộp'. The unit is dropped; read the unit column."""
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return float(x) if not (isinstance(x, float) and math.isnan(x)) else default
    m = re.match(r"\s*(-?\d+(?:[.,]\d+)?)", str(x)) if x is not None else None
    return float(m[1].replace(",", ".")) if m else default


def _map_unique(series, fn) -> pd.Series:
    """Apply fn once per distinct value (date / money columns repeat a few hundred values over 100k rows)."""
    s = pd.Series(series)
    try:
        uniq = pd.unique(s.dropna())
        table = {v: fn(v) for v in uniq}
        return s.map(lambda v: table.get(v, fn(v)) if v is not None and v == v else fn(v))
    except TypeError:  # unhashable cells
        return s.map(fn)


def money(series) -> pd.Series:
    """Vectorised to_number for a column."""
    return _map_unique(series, to_number)


# ---------------------------------------------------------------- dates

_EXCEL_ORIGIN = pd.Timestamp("1899-12-30")


def year_of(name) -> int | None:
    """Year written in a sheet or file name: 'T04-2025', '07.2026', 'BanHang_2025.xlsx' -> 2025/2026."""
    m = re.search(r"(20\d{2})", str(name or ""))
    return int(m[1]) if m else None


def to_datetime(x, year: int | None = None):
    """Date/time cell -> pandas.Timestamp (NaT if unreadable). Handles datetime objects, Excel serials
    (45658), ISO '2025-02-01 06:03:42' (never day-first), '01/04/2025', '1-4-2025', '1.4.2025',
    '1/4' or '1 thg 4' (needs `year`, e.g. year_of(sheet_name)), '1 thg 6 2025', 'ngày 1 tháng 6 năm 2025'."""
    if x is None or isinstance(x, bool):
        return pd.NaT
    if isinstance(x, (pd.Timestamp, _dt.datetime, _dt.date)):
        return pd.Timestamp(x)
    if isinstance(x, (int, float)):
        if isinstance(x, float) and math.isnan(x):
            return pd.NaT
        return _EXCEL_ORIGIN + pd.Timedelta(days=float(x)) if 20000 <= x <= 80000 else pd.NaT
    s = fold(x)
    if not s:
        return pd.NaT
    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})(?:[ t](\d{1,2}):(\d{2})(?::(\d{2}))?.*)?", s)
    if m:
        return _mk(int(m[1]), int(m[2]), int(m[3]), m[4], m[5], m[6])
    s = re.sub(r"^ngay\s*", "", s)
    m = re.fullmatch(r"(\d{1,2})\s*(?:thg|thang|/|-|\.)\s*(\d{1,2})(?:\s*(?:nam|/|-|\.|\s)\s*(\d{2,4}))?"
                     r"(?:\s+(\d{1,2}):(\d{2})(?::(\d{2}))?)?", s)
    if m:
        y = int(m[3]) if m[3] else year
        if y is None:
            return pd.NaT
        y = y + 2000 if y < 100 else y
        return _mk(y, int(m[2]), int(m[1]), m[4], m[5], m[6])
    if re.fullmatch(r"\d{5}(\.\d+)?", s):
        return to_datetime(float(s))
    return pd.NaT


def _mk(y, mo, d, hh=None, mm=None, ss=None):
    try:
        return pd.Timestamp(y, mo, d, int(hh or 0), int(mm or 0), int(ss or 0))
    except ValueError:
        return pd.NaT


def to_date(x, year: int | None = None):
    """Like to_datetime but drops the time: Timestamp at midnight."""
    t = to_datetime(x, year)
    return t.normalize() if not pd.isna(t) else pd.NaT


def dates(series, year: int | None = None) -> pd.Series:
    """Vectorised to_date for a column."""
    return _map_unique(series, lambda v: to_date(v, year))


# ---------------------------------------------------------------- order tables

def ffill_orders(df: pd.DataFrame, order_col: str, cols: list[str] | None = None,
                 min_blank: float = 0.9) -> pd.DataFrame:
    """Lines whose order id is blank belong to the order above: fill the id and the order-level `cols`
    down from the order's first line. Default `cols`: columns blank on at least `min_blank` of the
    follow-up lines (a few stray values on follow-up lines must not stop the fill). Returns a copy."""
    out = df.copy()
    blank = out[order_col].isna() | (out[order_col].astype(str).str.strip() == "")
    group = (~blank).cumsum()
    if cols is None:
        cols = [c for c in out.columns if blank.any() and out.loc[blank, c].isna().mean() >= min_blank]
    for c in set(cols) | {order_col}:
        out[c] = out.groupby(group)[c].transform("first")
    return out


def dedupe_order_lines(df: pd.DataFrame, order_col: str) -> pd.DataFrame:
    """Drop lines identical in every cell within the same order (export duplicates). Call after ffill_orders."""
    return df.drop_duplicates(keep="first").reset_index(drop=True) if order_col in df else df


# ---------------------------------------------------------------- calendar

def lunar_to_solar(calendar: pd.DataFrame, lunar_day: int, lunar_month: int, lunar_year: int,
                   leap: bool = False) -> pd.Timestamp:
    """Solar date of a lunar date, from calendar.csv. NaT if outside the calendar."""
    c = calendar
    lp = c["lunar_leap"].astype(str).str.lower().isin(["true", "1"])
    hit = c[(c.lunar_day == lunar_day) & (c.lunar_month == lunar_month) & (c.lunar_year == lunar_year) & (lp == leap)]
    return pd.Timestamp(hit.iloc[0]["date"]) if len(hit) else pd.NaT


# parse_* are the names prompts from v3 on use: v2's logs showed the model mixing our to_datetime
# up with pandas' (pd.to_datetime(..., year=2026)). The to_* names stay for the v1/v2 prompts.
parse_number, parse_qty, parse_money = to_number, to_qty, money
parse_date, parse_datetime, parse_dates = to_date, to_datetime, dates

EXPORTS = {name: globals()[name] for name in [
    "fold", "same_text", "to_number", "to_qty", "money", "year_of", "to_datetime", "to_date", "dates",
    "ffill_orders", "dedupe_order_lines", "lunar_to_solar",
    "parse_number", "parse_qty", "parse_money", "parse_date", "parse_datetime", "parse_dates"]}

DOC = """# Helpers already defined (use them instead of writing your own parsers)
- to_number(x) -> float|nan: money/number cells: 55000, '55.000đ', '35,000', '55k', '2,825tr', '1tr312'(=1312000), '2 tỷ'. money(series) applies it to a column.
- to_qty(x) -> float|nan: quantity cells: 2, '1 bình', '2,5 kg' (unit dropped; the unit is in its own column).
- to_date(x, year=None) / to_datetime(x, year=None) -> Timestamp|NaT: datetime objects, Excel serials, ISO (never day-first), '01/04/2025', '1 thg 6 2025', and '1/4' or '1 thg 4' when you pass year (year_of(sheet_name) reads it from 'T04-2025', '07.2026'). dates(series, year=None) applies to_date to a column.
- fold(s): lowercase, no accents, single spaces, for matching names typed without diacritics. same_text(a, b): equal up to case/accents/spacing/punctuation.
- ffill_orders(df, order_col, cols=None): fill the order id and order-level cells down onto follow-up lines of the same order. dedupe_order_lines(df, order_col): drop exact duplicate lines (call after ffill_orders).
- lunar_to_solar(calendar, lunar_day, lunar_month, lunar_year, leap=False) -> Timestamp: solar date of a lunar date.
"""

DOC_V3 = """# Helpers already defined (only for raw cells that have no parsed column)
- parse_number(x) -> float|nan: money/number cells: 55000, '55.000đ', '35,000', '55k', '2,825tr', '1tr312'(=1312000), '2 tỷ'. parse_money(series) applies it to a column.
- parse_qty(x) -> float|nan: quantity cells: 2, '1 bình', '2,5 kg' (unit dropped; the unit is in its own column).
- parse_date(x, year=None) / parse_datetime(x, year=None) -> Timestamp|NaT: every date spelling in these files; pass year (year_of(table_key)) for '1/4'. parse_dates(series, year=None) applies parse_date to a column.
- fold(s): lowercase, no accents, single spaces, for matching names typed without diacritics. same_text(a, b): equal up to case/accents/spacing/punctuation.
- dedupe_order_lines(df, order_col): drop lines identical in every cell within the same order.
- lunar_to_solar(calendar, lunar_day, lunar_month, lunar_year, leap=False) -> Timestamp: solar date of a lunar date.
"""
