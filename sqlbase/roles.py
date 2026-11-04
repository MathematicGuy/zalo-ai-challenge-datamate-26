"""Two fixed roles around the SQL writer; neither loops. The reader picks the rulebook lines a question depends on, which then sit right
before the question. The reviewer, when the samples disagree, sees the candidate queries with their results and picks one or rejects them all."""
import hashlib
import json
import re

MAX_LINES = 8

READER_SYSTEM = """You read a shop's business rulebook for a colleague who will write the SQL. Given one Vietnamese question, pick the numbered lines needed to answer it correctly: the definition of each business word in the question, which orders count, how the period and any ranking or tie are handled, rounding and answer format, and any rule for this shop's industry. Do not answer the question. Pick at most 8 lines, the fewer the better.
Reply with exactly one line and nothing else:
LINES: n, n, n"""

REVIEW_SYSTEM = """You review candidate answers to one Vietnamese question about one shop's data. Several queries were written and run, and they disagree. Decide which one follows the question and the RULEBOOK.

Check each candidate in this order:
1. DATA: the right tables or views for what is asked.
2. PERIOD: exactly the period asked, taken from DATE ANCHORS.
3. FILTERS: which orders count, and every value the question names.
4. COUNT: what is counted or summed, distinct ids or quantities, and the units.
5. ANSWER: the result has the type, shape and rounding the question asks for.

Choose a candidate only if it passes all five. If every candidate has a fault, or you cannot tell, choose NONE: a wrong answer costs more than no answer.
End your reply with exactly one line: CHOICE: A (or B, C, NONE)"""


def number_lines(rulebook):
    """[(section heading, line)] for each line a reader may pick, and the rulebook text with those lines numbered from 1."""
    rows, text, head = [], [], ""
    for raw in rulebook.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            head = line.lstrip("# ").strip()
            text.append(line)
        else:
            rows.append((head, line))
            text.append(f"[{len(rows)}] {line}")
    return rows, "\n".join(text)


def reader_chat(numbered, readme, question):
    return [{"role": "system", "content": READER_SYSTEM + "\n\nRULEBOOK:\n" + numbered},
            {"role": "user", "content": "THIS SHOP:\n" + readme[:1500] + "\n\nQUESTION: " + question}]


def parse_lines(text, n):
    """Valid line numbers from the reader's reply, in its order; nothing when it did not follow the format."""
    m = re.search(r"LINES?\s*:\s*(.*)", text)
    if not m:
        return []
    picks = [x for x in dict.fromkeys(int(v) for v in re.findall(r"\d+", m.group(1))) if 1 <= x <= n]
    return picks[:MAX_LINES]


def rule_block(rows, picks):
    if not picks:
        return ""
    return "RULES FOR THIS QUESTION (copied from the RULEBOOK; apply them):\n" + "\n".join(f"- [{rows[p - 1][0]}] {rows[p - 1][1]}" for p in picks)


def show(pred, limit=700):
    """A candidate's result as the reviewer sees it."""
    a = pred.get("answer")
    if pred["type"] == "table" and isinstance(a, dict):
        rows = a.get("rows", [])
        a = {"columns": a.get("columns"), "rows": rows[:8], "total_rows": len(rows)}
    elif pred["type"] == "list" and isinstance(a, list):
        a = {"items": a[:15], "total_items": len(a)}
    return f"{pred['type']}: " + json.dumps(a, ensure_ascii=False, default=str)[:limit]


def candidates(question, samples, key, parse, pick, below=0.75, top=3):
    """Up to `top` answer groups as labeled candidates when the largest holds under `below` of the samples; none when they agree.
    samples: [(text, pred)]; key groups preds as the vote does; pick chooses a group's answer. Labels follow a hash of the question and
    vote counts stay hidden, so neither position nor majority steers the reviewer."""
    groups = {}
    for text, pred in samples:
        if pred:
            groups.setdefault(key(pred), []).append((text, pred))
    if not groups:
        return []
    ranked = sorted(groups.values(), key=len, reverse=True)
    if len(ranked[0]) / max(1, len(samples)) >= below:
        return []
    chosen = ranked[:top]
    order = sorted(range(len(chosen)), key=lambda i: hashlib.sha256(f"{question}|{i}".encode()).hexdigest())
    out = []
    for label, i in zip("ABC", order):
        g = chosen[i]
        sql = next((s for s in (parse(t.split("\n#REPAIR\n")[-1])[1] for t, _ in g) if s), None)
        out.append({"label": label, "pred": pick([p for _, p in g]), "sql": sql, "votes": len(g)})
    return out


def review_chat(system, user, cands):
    """system: REVIEW_SYSTEM + rulebook. user: the writer's user message, so the reviewer reads the same schema, anchors and rules."""
    body = "\n\n".join(f"CANDIDATE {c['label']}\n```sql\n{c['sql'] or '(no query)'}\n```\nRESULT: {show(c['pred'])}" for c in cands)
    return [{"role": "system", "content": system}, {"role": "user", "content": f"{user}\n\n{body}"}]


def parse_choice(text, cands):
    """A candidate's label or "NONE"; None when the reply was cut off or broke the format, so the vote stands."""
    if "</think>" not in text:
        return None
    m = re.findall(r"CHOICE\s*:\s*\**\s*(NONE|[A-C])\b", text.split("</think>")[-1])
    if not m:
        return None
    return m[-1] if m[-1] == "NONE" or any(c["label"] == m[-1] for c in cands) else None
