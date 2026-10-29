"""Value linking: names in the data (products, customers, groups...) whose words appear in the
question, listed next to it so the model filters on a real spelling instead of guessing one
(CHESS-style value retrieval). Computed from the shop's own tables only."""
from __future__ import annotations

import re

from ..tables import Business
from ..text import fold
from .rules import TOKEN, tokens


class ValueIndex:
    """Distinct short string values of a business, by folded token set."""

    def __init__(self, biz: Business, max_distinct: int = 3000, max_len: int = 80):
        self.entries: list[tuple[str, str, str, frozenset]] = []   # (value, table, column, tokens)
        seen = set()
        for name, df in biz.frames.items():
            for c in df.columns:
                col = df[c].dropna()
                if col.empty or not isinstance(col.iloc[0], str):
                    continue
                vals = col.astype(str).unique()
                if len(vals) > max_distinct:
                    continue
                for v in vals:
                    if len(v) > max_len or not re.search(r"[A-Za-zÀ-ỹ]", v):
                        continue
                    key = (fold(v), str(c))
                    if key in seen:
                        continue
                    seen.add(key)
                    toks = frozenset(TOKEN.findall(fold(v)))
                    if toks:
                        self.entries.append((v, name, str(c), toks))

    def match(self, question: str, k: int = 8) -> list[tuple[str, str, str]]:
        """Values all of whose words occur in the question, or names sharing ≥ 3 content words and ≥ 2/3
        of their words with it (filler like 'bao nhiêu' must not make 'La Bảo Hoàng' a match)."""
        q_all = set(TOKEN.findall(fold(question)))
        q_content = set(tokens(question))
        hits = []
        for v, table, col, toks in self.entries:
            full = toks <= q_all
            inter = len(toks & q_content)
            if full or (inter >= 3 and inter / len(toks) >= 2 / 3):
                hits.append((len(toks) if full else inter, len(toks), v, table, col))
        hits.sort(key=lambda h: (-h[0], h[1]))
        out, seen = [], set()
        for _, _, v, table, col in hits:
            if (v, col) not in seen:
                seen.add((v, col))
                out.append((v, table, col))
            if len(out) == k:
                break
        return out


def value_tokens(hits: list[tuple[str, str, str]]) -> frozenset[str]:
    return frozenset(t for v, _, _ in hits for t in TOKEN.findall(fold(v)))


def format_values(hits: list[tuple[str, str, str]]) -> str:
    return "\n".join(f"- {v!r} in tables[{t!r}][{c!r}]" for v, t, c in hits)
