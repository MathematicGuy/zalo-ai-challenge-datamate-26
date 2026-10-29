import datetime as dt
import math

import pandas as pd
import pytest

from datamate.helpers import (dedupe_order_lines, ffill_orders, fold, lunar_to_solar, same_text, to_date,
                              to_datetime, to_number, to_qty, year_of)

from .conftest import TRAIN


# Spellings seen in the released data (EDA section 6) plus a few neighbours.
@pytest.mark.parametrize("raw, want", [
    (55000, 55000), (55000.0, 55000), ("55000", 55000), ("55.000đ", 55000), ("55.000 đ", 55000),
    ("1.500đ", 1500), ("35,000", 35000), ("1,234,567", 1234567), ("55k", 55000), ("7,5k", 7500),
    ("2,825tr", 2825000), ("332,5972tr", 332597200), ("1tr312", 1312000), ("409tr663", 409663000),
    ("5tr520", 5520000), ("2tr5", 2500000), ("1,2 triệu", 1200000), ("2 tỷ", 2e9), ("-12.000", -12000),
    ("18.6", 18.6), ("7,5", 7.5), ("1 bình", 1), ("4417k", 4417000), ("100k", 100000), ("1tr", 1e6),
])
def test_to_number(raw, want):
    assert to_number(raw) == pytest.approx(want)


@pytest.mark.parametrize("raw", [None, "", "abc", float("nan"), True])
def test_to_number_unreadable(raw):
    assert math.isnan(to_number(raw))


@pytest.mark.parametrize("raw, want", [(2, 2), ("2", 2), ("1 bình", 1), ("2,5 kg", 2.5), ("10 hộp", 10)])
def test_to_qty(raw, want):
    assert to_qty(raw) == want


@pytest.mark.parametrize("raw, year, want", [
    (dt.datetime(2025, 4, 1, 6, 4, 59), None, "2025-04-01 06:04:59"),
    (45658, None, "2025-01-01 00:00:00"),
    ("2025-02-01 06:03:42", None, "2025-02-01 06:03:42"),   # ISO is never day-first
    ("01/04/2025", None, "2025-04-01 00:00:00"),
    ("1/4", 2025, "2025-04-01 00:00:00"),
    ("1 thg 6 2025", None, "2025-06-01 00:00:00"),
    ("1 thg 1", 2026, "2026-01-01 00:00:00"),
    ("15-08-2026", None, "2026-08-15 00:00:00"),
    ("ngày 3 tháng 2 năm 2026", None, "2026-02-03 00:00:00"),
    ("01/04/2025 13:05", None, "2025-04-01 13:05:00"),
])
def test_to_datetime(raw, year, want):
    assert str(to_datetime(raw, year)) == want


def test_dates_without_year_are_nat():
    assert pd.isna(to_date("1/4")) and pd.isna(to_date("31/02/2025")) and pd.isna(to_date("hôm qua"))


def test_year_of():
    assert (year_of("T04-2025"), year_of("07.2026"), year_of("BanHang_2025.xlsx::Sheet1"), year_of("Sheet1")) == \
           (2025, 2026, 2025, None)


def test_text():
    assert fold("  Trà  Sữa Ô LONG (L) ") == "tra sua o long (l)" and fold("Đường") == "duong"
    assert same_text("Sting dâu 330ml", "sting dau 330 ML") and not same_text("Coca 330ml", "Coca 390ml")


def test_ffill_then_dedupe_keeps_identical_lines_of_different_orders():
    df = pd.DataFrame({"ma": ["A", None, "B", None, "B"],
                       "khach": ["Lan", None, "Tú", None, "Tú"],
                       "hang": ["x", "y", "x", "y", "x"], "sl": [1, 1, 1, 1, 1]})
    f = ffill_orders(df, "ma")
    assert f.ma.tolist() == ["A", "A", "B", "B", "B"] and f.khach.tolist() == ["Lan", "Lan", "Tú", "Tú", "Tú"]
    d = dedupe_order_lines(f, "ma")
    # B's repeated 'x' line is an export duplicate; A's 'y' and B's 'y' are different sales.
    assert len(d) == 4 and (d.hang == "y").sum() == 2


def test_lunar_to_solar_matches_known_tet():
    cal = pd.read_csv(TRAIN / "calendar.csv")
    assert str(lunar_to_solar(cal, 1, 1, 2025).date()) == "2025-01-29"   # Tết Ất Tỵ
    assert str(lunar_to_solar(cal, 1, 1, 2026).date()) == "2026-02-17"   # Tết Bính Ngọ


def test_ffill_tolerates_stray_values_on_follow_up_lines():
    ids = ["A"] + [None] * 10 + ["B"] + [None] * 10
    cust = ["Lan"] + [None] * 9 + ["Lan"] + ["Tú"] + [None] * 10      # one stray name on A's follow-ups
    f = ffill_orders(pd.DataFrame({"ma": ids, "khach": cust, "hang": list("xyzxyzxyzxyzxyzxyzxyzx")}), "ma")
    assert f.khach.tolist() == ["Lan"] * 11 + ["Tú"] * 11
