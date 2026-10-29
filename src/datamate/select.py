"""Pick one answer from k executed samples by agreement, and decide whether to abstain.

Samples are clustered by the value the scorer would compare (not by code text), the largest cluster
wins, and its share of the k samples is the confidence. Under PA (+1 / 0 / -1) answering pays only
when P(correct) > 0.5, so the abstain threshold is tuned on validation against that criterion.
"""
from __future__ import annotations

from collections import Counter

from .text import norm_answer


def _cell(v):
    return round(float(v), 2) if isinstance(v, (int, float)) and not isinstance(v, bool) else norm_answer(v)


def answer_key(ans: dict) -> tuple:
    t = ans["type"]
    if t in ("unanswerable", "abstain"):
        return (t,)
    v = ans["answer"]
    if t in ("number", "percent"):
        return (t, round(float(v), 2))
    if t == "string":
        return (t, norm_answer(v))
    if t == "list":
        return (t, tuple(_cell(x) for x in v))
    if t == "table":
        return (t, len(v["columns"]), frozenset(tuple(_cell(x) for x in r) for r in v["rows"]))
    return (t, v)


def vote(answers: list[dict | None], k: int) -> dict:
    """answers: one normalized answer (or None for a failed sample) per sample."""
    valid = [a for a in answers if a is not None and a["type"] != "abstain"]
    if not valid:
        return {"answer": None, "support": 0.0, "clusters": []}
    groups: dict[tuple, list[dict]] = {}
    for a in valid:
        groups.setdefault(answer_key(a), []).append(a)
    ranked = sorted(groups.values(), key=len, reverse=True)
    best = ranked[0]
    # Representative: the most frequent exact rendering inside the winning cluster.
    rep = Counter(repr(a) for a in best).most_common(1)[0][0]
    answer = next(a for a in best if repr(a) == rep)
    return {"answer": answer, "support": len(best) / k,
            "clusters": [{"answer": g[0], "size": len(g)} for g in ranked]}


def decide(voted: dict, tau: float) -> dict:
    """Final submission row body: the winning answer if its support reaches tau, else abstain."""
    if voted["answer"] is not None and voted["support"] >= tau:
        return voted["answer"]
    return {"type": "abstain"}
