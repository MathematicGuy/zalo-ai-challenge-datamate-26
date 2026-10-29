"""Vietnamese text helpers shared by every layer (one `fold`, one answer normalization)."""
from __future__ import annotations

import math
import re
import unicodedata


def norm_answer(s) -> str:
    """The organizers' string normalization: NFC, lowercase, non letter/digit/space -> space, collapse."""
    s = unicodedata.normalize("NFC", str(s)).lower()
    s = "".join(c if c.isalnum() or c.isspace() else " " for c in s)
    return " ".join(s.split())


def deaccent(s) -> str:
    s = unicodedata.normalize("NFD", str(s).replace("đ", "d").replace("Đ", "D"))
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def has_diacritics(s: str) -> bool:
    return deaccent(s) != s


def fold(s) -> str:
    """Lowercase, no accents (đ->d), single spaces: a matching key for text typed with or without
    diacritics. Never use it to produce an answer. None / NaN fold to ''."""
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return ""
    return re.sub(r"\s+", " ", deaccent(s).lower()).strip()
