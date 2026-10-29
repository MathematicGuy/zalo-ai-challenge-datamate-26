import pytest

from datamate.normalize import expand_abbreviations


@pytest.mark.parametrize("raw, want", [
    ("dthu hqua cua cua hang la bn z", "doanh thu hôm qua cua cua hang la bao nhiêu vậy"),
    ("Dthu th1/2026 đc bnhiu z?", "doanh thu tháng 1/2026 được bao nhiêu vậy?"),
    ("T10/2025 dthu hơn hay kém", "tháng 10/2025 doanh thu hơn hay kém"),
    ("3/26 thợ mô làm nhiều nhất rứa?", "3/26 thợ nào làm nhiều nhất vậy?"),
    ("bình quân mỗi o đờ", "bình quân mỗi đơn"),
])
def test_expands(raw, want):
    assert expand_abbreviations(raw) == want


@pytest.mark.parametrize("text", [
    "check giùm", "hơn hay kém", "biển số 59-K1", "giá 55k", "mua nhiu đơn nhất",
    "Ống PVC Bình Minh D50 cây 4m", "Topping Thạch cà phê", "Size XL",
])
def test_leaves_alone(text):
    assert expand_abbreviations(text) == text
