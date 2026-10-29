"""Penalized Accuracy, implemented from the Evaluation tab (no official scorer was released).

Per question: +1 correct, 0 abstain or missing, -1 wrong. An answer is correct only if its type equals
the reference type and its value matches under that type's rule.
"""
from __future__ import annotations

import math
from collections import Counter

from .text import norm_answer

VALUE_TYPES = {"number", "percent", "string", "date", "boolean", "list", "table"}
ALL_TYPES = VALUE_TYPES | {"unanswerable", "abstain"}


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _num_eq(a, b) -> bool:
    return _is_num(a) and _is_num(b) and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-6)


def _pct_eq(a, b) -> bool:
    return _is_num(a) and _is_num(b) and math.isclose(round(a + 0.0, 2), round(b + 0.0, 2), abs_tol=1e-9)


def _str_eq(pred, forms) -> bool:
    return isinstance(pred, str) and norm_answer(pred) in {norm_answer(f) for f in forms}


def _cell_eq(pred, ref, alias=()) -> bool:
    """A list element or table cell: numbers by the number rule, names by the string rule."""
    if _is_num(ref):
        return _num_eq(pred, ref)
    return _str_eq(pred, [ref, *alias])


def _match_list(pred, ref, alias) -> bool:
    if not isinstance(pred, list) or len(pred) != len(ref):
        return False
    alias = alias or [[] for _ in ref]
    return all(_cell_eq(p, r, a) for p, r, a in zip(pred, ref, alias))


def _match_table(pred, ref, alias) -> bool:
    if not (isinstance(pred, dict) and isinstance(pred.get("columns"), list) and isinstance(pred.get("rows"), list)):
        return False
    rcols, pcols = ref["columns"], pred["columns"]
    if len(pcols) != len(rcols) or any(not isinstance(r, list) or len(r) != len(rcols) for r in pred["rows"]):
        return False
    # Columns by normalized name; if the names do not match, by position.
    pnorm = [norm_answer(c) for c in pcols]
    if sorted(pnorm) == sorted(norm_answer(c) for c in rcols) and len(set(pnorm)) == len(pnorm):
        order = [pnorm.index(norm_answer(c)) for c in rcols]
    else:
        order = list(range(len(rcols)))
    prows = [[row[i] for i in order] for row in pred["rows"]]
    rrows = ref["rows"]
    if len(prows) != len(rrows):
        return False
    alias = alias or [[] for _ in rrows]
    # Rows as a set: greedy one-to-one matching (rows are distinct objects, so greedy is exact enough).
    unused = list(range(len(prows)))
    for rrow, ralias in zip(rrows, alias):
        hit = next((j for j in unused if all(_cell_eq(p, r, ralias if k == 0 else ())
                                             for k, (p, r) in enumerate(zip(prows[j], rrow)))), None)
        if hit is None:
            return False
        unused.remove(hit)
    return True


def score_one(pred: dict | None, ref: dict) -> int:
    if pred is None or pred.get("type") in (None, "abstain"):
        return 0
    ptype, rtype = pred["type"], ref["type"]
    if ptype != rtype:
        return -1
    if rtype == "unanswerable":
        return 1
    p, r, alias = pred.get("answer"), ref["answer"], ref.get("alias") or []
    ok = {
        "number": lambda: _num_eq(p, r),
        "percent": lambda: _pct_eq(p, r),
        "string": lambda: _str_eq(p, [r, *alias]),
        "date": lambda: isinstance(p, str) and p == r,
        "boolean": lambda: isinstance(p, bool) and p == r,
        "list": lambda: _match_list(p, r, alias),
        "table": lambda: _match_table(p, r, alias),
    }[rtype]()
    return 1 if ok else -1


def penalized_accuracy(preds: dict[str, dict], refs: dict[str, dict]) -> dict:
    """preds/refs keyed by question id. Questions missing from preds count as abstain."""
    s = {qid: score_one(preds.get(qid), ref) for qid, ref in refs.items()}
    c = Counter(s.values())
    n = len(refs)
    return {"PA": (c[1] - c[-1]) / n if n else 0.0, "n": n, "correct": c[1], "wrong": c[-1], "abstain": c[0],
            "precision": c[1] / (c[1] + c[-1]) if c[1] + c[-1] else float("nan"), "per_question": s}
