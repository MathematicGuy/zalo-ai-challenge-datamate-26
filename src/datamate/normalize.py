"""Expand Vietnamese chat abbreviations in a question.

General language knowledge only (how people type in chat), not statistics of any split. The model
always sees the original question too; the expansion is a reading aid, e.g. 'hqua' is 'hôm qua'
(yesterday), which v1 misread as 'last month'.
"""
from __future__ import annotations

import re

# (pattern, replacement); patterns are case-insensitive and bounded by non-letters.
_RULES = [
    (r"dthu|d\.thu|doanh thuu", "doanh thu"),
    (r"hqua|hom qua|hôm wa|hqa", "hôm qua"),
    (r"hnay|hom nay|bữa nay|bua nay", "hôm nay"),
    # Not bare 'nhiu' / 'nhiêu': 'mua nhiu đơn nhất' means 'nhiều' (most), expanding it flips the question.
    (r"bnhiu|bnhieu|bnh|bn|bao nhiu|bao nhieu|bao nhiêu", "bao nhiêu"),
    (r"tháng ni|thang ni", "tháng này"),
    (r"tuần trc|tuan trc|tuần rùi|tuan rui", "tuần trước"),
    (r"trc", "trước"),
    (r"sp", "sản phẩm"),
    (r"đc|dc|đk", "được"),
    (r"ko|k|hok|hông|hong|kh0", "không"),
    (r"mik|mk|mjk", "mình"),
    (r"kua|của", "của"),
    (r"kó|ko' có", "có"),
    (r"cko|ch0", "cho"),
    (r"jùm|giùm|dùm|gium|jum", "giùm"),
    (r"nhứt|nhất", "nhất"),
    (r"ncc", "nhà cung cấp"),
    (r"nv", "nhân viên"),
    (r"sl", "số lượng"),
    (r"o đờ|ô đờ|order|bill", "đơn"),
    (r"vs", "với"),
    (r"z|zậy|dị|rứa", "vậy"),
    (r"mô", "nào"),
    (r"răng", "sao"),
]
# Group each alternation so both boundaries apply to every alternative; a boundary is any non-word
# character, so 'check', 'kém' and plate '59-K1' are left alone.
_COMPILED = [(re.compile(rf"(?<!\w)(?:{p})(?!\w)", re.I), r) for p, r in _RULES]
# 'th5/2026', 'T10/2025', 't 12' -> 'tháng 5/2026'; the month number follows directly.
_MONTH = re.compile(r"(?<!\w)(?:th|t)\s?(?=\d{1,2}(?:\s*[/\-]\s*\d{2,4}|\b))", re.I)


def expand_abbreviations(q: str) -> str:
    out = _MONTH.sub("tháng ", q)
    for rx, rep in _COMPILED:
        out = rx.sub(rep, out)
    return re.sub(r"\s+", " ", out).strip()
