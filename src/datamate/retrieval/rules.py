"""Rulebook-row retrieval behind one interface: lexical, bi-encoder, and their RRF hybrid.

Measured on 48 hand-labelled train questions (notebooks/03_retrieval): lexical R@5 0.47 / P@3 0.30,
off-the-shelf bi-encoders lower (Qwen3-Embedding-0.6B R@5 0.31, bge-m3 0.36), hybrid ≈ lexical.
At that precision two of three injected rows are irrelevant, and irrelevant context distracts
(Shi et al. 2023), so the agent does not inject rules by default (Config.rules_k = 0).
"""
from __future__ import annotations

import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

from ..normalize import expand_abbreviations
from ..text import fold

TOKEN = re.compile(r"[a-z0-9]+")
# Question filler that carries no rule meaning (folded). Time words (thang, tuan, hom, qua, nay, ngay,
# nam, tu, den, het) are NOT here: they select the §2 period rules (dropping them cost R@5 0.47 -> 0.40).
STOP = set("""la cua va co cho bao nhieu bn bnhiu nhiu may gi nao the thi nhe a ha z vay nha oi e em anh chi minh
toi ban giup xem check coi giu gium jum dc duoc da dang trong nhung cac mot""".split())


def tokens(s: str) -> list[str]:
    """Content words of a folded text: no filler, no single letters, no tokens with digits."""
    return [t for t in TOKEN.findall(fold(s)) if t not in STOP and len(t) > 1 and not any(c.isdigit() for c in t)]


@dataclass(frozen=True)
class Rule:
    section: str
    name: str
    text: str


def parse_rules(rulebook: str) -> list[Rule]:
    """Every table row of the rulebook (headers and separator rows skipped)."""
    rules, section = [], ""
    for line in rulebook.splitlines():
        if line.startswith("#"):
            section = line.lstrip("# ").strip()
            continue
        m = re.match(r"^\|\s*([^|]+?)\s*\|\s*(.+?)\s*\|\s*$", line)
        if m and not set(m[1]) <= set("-: ") and fold(m[1]) not in ("luat", "cach noi", "cach ghi", "tu", "type"):
            rules.append(Rule(section, m[1], m[2]))
    return rules


def format_rules(rules: list[Rule]) -> str:
    return "\n".join(f"- [{r.section}] {r.name}: {r.text}" for r in rules)


class RuleRetriever(Protocol):
    rules: list[Rule]

    def rank(self, question: str, exclude: frozenset[str] = frozenset()) -> list[int]:
        """Indices into `rules`, best first. `exclude`: words of data values the question names."""


def top_rules(retriever: RuleRetriever, question: str, k: int, exclude=frozenset()) -> list[Rule]:
    return [retriever.rules[i] for i in retriever.rank(question, exclude)[:k]]


class LexicalRuleRetriever:
    """IDF-weighted word overlap; rule names count three times their text."""

    def __init__(self, rulebook: str):
        self.rules = parse_rules(rulebook)
        self.docs = [Counter(tokens(r.name) * 3 + tokens(r.text)) for r in self.rules]
        df = Counter(t for d in self.docs for t in d)
        self.idf = {t: math.log((len(self.docs) + 1) / (c + 0.5)) for t, c in df.items()}

    def rank(self, question: str, exclude: frozenset[str] = frozenset()) -> list[int]:
        # Rules are about concepts: a rare word inside a product name ('Bình Minh') must not pull in 'trung bình'.
        q = set(tokens(expand_abbreviations(question))) - set(exclude)
        scored = [(sum(self.idf.get(t, 0) * min(d[t], 3) for t in q), i) for i, d in enumerate(self.docs)]
        return [i for s, i in sorted(scored, key=lambda x: (-x[0], x[1])) if s > 0]


# Query instruction for instruction-tuned embedders (Qwen3-Embedding); bge-m3 takes plain text.
QUERY_INSTRUCT = ("Instruct: Given a shop owner's question in Vietnamese, retrieve the business rule that "
                  "defines how to compute the answer\nQuery: ")


@lru_cache(maxsize=2)
def _load_embedder(name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name, device=os.environ.get("DATAMATE_EMB_DEVICE", "cpu"))


class DenseRuleRetriever:
    """Bi-encoder cosine similarity. Its weights count toward the 15B inference budget."""

    def __init__(self, rulebook: str, model: str = "Qwen/Qwen3-Embedding-0.6B"):
        self.rules = parse_rules(rulebook)
        self.model = model
        self._emb = _load_embedder(model)
        self._doc = self._emb.encode([f"{r.section} | {r.name}: {r.text}" for r in self.rules],
                                     normalize_embeddings=True, batch_size=16, show_progress_bar=False)

    def rank(self, question: str, exclude: frozenset[str] = frozenset()) -> list[int]:
        text = expand_abbreviations(question)
        if "qwen3-embedding" in self.model.lower():
            text = QUERY_INSTRUCT + text
        sims = self._emb.encode([text], normalize_embeddings=True, show_progress_bar=False)[0] @ self._doc.T
        return sorted(range(len(self.rules)), key=lambda i: -sims[i])


def rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Reciprocal rank fusion: combines rankings by position, so their scores need no calibration."""
    score: dict[int, float] = {}
    for ranking in rankings:
        for pos, item in enumerate(ranking):
            score[item] = score.get(item, 0.0) + 1.0 / (k + pos + 1)
    return sorted(score, key=lambda i: (-score[i], i))


class HybridRuleRetriever:
    """RRF over several retrievers built on the same rulebook."""

    def __init__(self, retrievers: list[RuleRetriever]):
        self.parts = retrievers
        self.rules = retrievers[0].rules

    def rank(self, question: str, exclude: frozenset[str] = frozenset()) -> list[int]:
        return rrf([r.rank(question, exclude) for r in self.parts])


def make_rule_retriever(kind: str, rulebook: str) -> RuleRetriever:
    """'lexical' | 'dense' | 'dense:<hf model id>' | 'hybrid' (lexical + Qwen3-Embedding + bge-m3)."""
    if kind == "lexical":
        return LexicalRuleRetriever(rulebook)
    if kind.startswith("dense"):
        _, _, model = kind.partition(":")
        return DenseRuleRetriever(rulebook, model or "Qwen/Qwen3-Embedding-0.6B")
    if kind == "hybrid":
        return HybridRuleRetriever([LexicalRuleRetriever(rulebook), DenseRuleRetriever(rulebook),
                                    DenseRuleRetriever(rulebook, "BAAI/bge-m3")])
    raise ValueError(f"unknown rule retriever {kind!r}")
