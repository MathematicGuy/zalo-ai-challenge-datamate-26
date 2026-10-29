"""From a model reply to a scorer-ready answer: extract the code, validate ANSWER, apply type guards."""
from __future__ import annotations

import math
import re

from .scoring import VALUE_TYPES
from .text import fold

_CODE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)
_PCT_CUE = re.compile(r"%|phan tram|ti le|ty le|ti trong|ty trong|chiem")


def extract_code(text: str) -> str | None:
    """The longest fenced python block of the final answer (text after </think> in thinking mode)."""
    text = text.rsplit("</think>", 1)[-1]
    blocks = _CODE.findall(text)
    if blocks:
        return max(blocks, key=len)
    return text if "ANSWER" in text else None


def _num(v):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ValueError(f"not a number: {v!r}")
    return int(round(v)) if abs(v - round(v)) < 1e-6 else round(float(v), 2)


def _cell(x):
    return x.strip() if isinstance(x, str) else _num(x)


def normalize_answer(ans) -> dict:
    """Validate the type and coerce the value to the one format the scorer accepts. Raises ValueError."""
    if not isinstance(ans, dict) or "type" not in ans:
        raise ValueError("ANSWER must be a dict with a 'type' key")
    t = ans["type"]
    if t == "unanswerable":
        return {"type": t, "reason": str(ans.get("reason", ""))[:300]}
    if t == "abstain":
        return {"type": t}
    if t not in VALUE_TYPES:
        raise ValueError(f"unknown type {t!r}")
    if "answer" not in ans or ans["answer"] is None:
        raise ValueError("missing 'answer'")
    v = ans["answer"]
    if t == "number":
        v = _num(v)
    elif t == "percent":
        v = round(float(_num(v)), 2)
    elif t == "string":
        if not isinstance(v, str):
            raise ValueError("string answer must be a str")
        v = v.strip()
    elif t == "date":
        if not (isinstance(v, str) and re.fullmatch(r"\d{4}-\d{2}(-\d{2})?", v)):
            raise ValueError(f"date must be YYYY-MM-DD or YYYY-MM, got {v!r}")
    elif t == "boolean":
        if not isinstance(v, bool):
            raise ValueError("boolean answer must be true/false")
    elif t == "list":
        if not isinstance(v, list) or not v:
            raise ValueError("list answer must be a non-empty list")
        v = [_cell(x) for x in v]
    elif t == "table":
        if not (isinstance(v, dict) and isinstance(v.get("columns"), list) and isinstance(v.get("rows"), list)):
            raise ValueError("table answer must be {'columns': [...], 'rows': [[...]]}")
        ncol = len(v["columns"])
        if any(not isinstance(r, list) or len(r) != ncol for r in v["rows"]):
            raise ValueError("every table row must have one cell per column")
        v = {"columns": [str(c) for c in v["columns"]], "rows": [[_cell(x) for x in r] for r in v["rows"]]}
    return {"type": t, "answer": v}


def postprocess(question: str, ans: dict | None) -> dict | None:
    """Deterministic type guard from the rulebook's answer formats (§9): a question asking for a
    percentage takes type percent. The model often returns the right value typed as number."""
    if ans is not None and ans.get("type") == "number" and _PCT_CUE.search(fold(question)):
        return {"type": "percent", "answer": round(float(ans["answer"]), 2)}
    return ans
