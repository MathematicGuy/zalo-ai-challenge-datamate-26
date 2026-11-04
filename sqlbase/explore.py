"""A bounded exploration agent for questions the samples disagree on: each reply is one action, code runs it and returns what it found, and the last reply must be final.
The model chooses what to look at; code keeps queries read-only and caps steps, rows and text."""
import difflib
import json
import re

from ingest import norm

ADDON = """

YOU MAY LOOK BEFORE YOU ANSWER. Work in steps; each reply holds one short reason line, then exactly one action:
ACTION: QUERY, then one ```sql``` block: runs a read-only query and shows the row count, columns and first rows. Use it to see what values a column holds, what a filter matches, or whether a partial result looks right.
ACTION: VALUES, then TERM: text: shows the values stored in this shop's data closest to the text.
ACTION: RULE, then TERM: text: shows the RULEBOOK lines mentioning the text.
ACTION: FINAL, then the TYPE line and one ```sql``` block: your answer. The query computes the answer.
Look only at what you are unsure of, and reply FINAL as soon as you are sure."""


def parse_action(text):
    """(kind, argument). A reply with a TYPE line and a query but no ACTION is treated as final, a bare query as a look."""
    t = text.split("</think>")[-1]
    m = re.findall(r"ACTION\s*:\s*\**\s*(QUERY|VALUES|RULE|FINAL)", t)
    sqls = re.findall(r"```(?:sql)?\s*(.*?)```", t, re.S)
    term = re.findall(r"TERM\s*:\s*(.+)", t)
    kind = m[-1] if m else ("FINAL" if sqls and re.search(r"TYPE:\s*[a-z]+", t) else "QUERY" if sqls else None)
    if kind == "QUERY":
        return (kind, sqls[-1].strip()) if sqls else (None, None)
    if kind in ("VALUES", "RULE"):
        return (kind, term[-1].strip()) if term else (None, None)
    return kind, t if kind else None


class Tools:
    def __init__(self, db, run_sql, known, rule_rows):
        self.db, self.run_sql, self.known, self.rule_rows = db, run_sql, known, rule_rows

    def query(self, sql):
        if re.search(r"(?i)\b(insert|update|delete|create|drop|alter|copy|attach|install|load|pragma)\b", sql) or not re.match(r"(?is)\s*(with|select|from)\b", sql):
            return "Only one read-only SELECT query is allowed."
        try:
            names, rows = self.run_sql(self.db, sql.rstrip().rstrip(";"))
        except Exception as e:
            return f"ERROR: {str(e)[:300]}"
        return f"{len(rows)} row(s), columns {json.dumps(names, ensure_ascii=False)}\nFirst rows: {json.dumps(rows[:8], ensure_ascii=False, default=str)[:1500]}"

    def values(self, term):
        t = norm(term)
        hits = [v for k, v in self.known.items() if t and t in k][:8]
        if len(hits) < 8:
            hits += [self.known[k] for k in difflib.get_close_matches(t, list(self.known), n=8 - len(hits), cutoff=0.6) if self.known[k] not in hits]
        return "Closest values: " + json.dumps(hits, ensure_ascii=False, default=str) if hits else "No stored value is close to that text."

    def rule(self, term):
        words = [w for w in norm(term).split("_") if len(w) >= 2]
        scored = []
        for head, line in self.rule_rows:
            n = norm(head + " " + line)
            s = sum(w in n for w in words)
            if s:
                scored.append((s, f"- [{head}] {line}"))
        scored.sort(key=lambda x: -x[0])
        return "\n".join(x[1] for x in scored[:6]) or "No rule line mentions that text."


def run_batch(generate, items, max_steps=4):
    """items: {key: (conversation, Tools)}; generate(conversations, keys) -> replies. Returns {key: {"final": text or None, "steps": [kinds], "conv": conversation}}."""
    state = {k: {"conv": list(c), "steps": [], "final": None, "tools": t} for k, (c, t) in items.items()}
    for step in range(1, max_steps + 1):
        active = [k for k, s in state.items() if s["final"] is None]
        if not active:
            break
        for k, reply in zip(active, generate([state[k]["conv"] for k in active], active)):
            s = state[k]
            kind, arg = parse_action(reply)
            s["conv"].append({"role": "assistant", "content": reply})
            s["steps"].append(kind or "BAD")
            if kind == "FINAL":
                s["final"] = reply
                continue
            if step == max_steps:
                continue
            seen = s["tools"].query(arg) if kind == "QUERY" else s["tools"].values(arg) if kind == "VALUES" else s["tools"].rule(arg) if kind == "RULE" else "Reply with exactly one ACTION."
            left = max_steps - step
            s["conv"].append({"role": "user", "content": seen + ("\nThis was your last look. Reply now with ACTION: FINAL, the TYPE line and one ```sql``` block." if left == 1 else f"\n{left - 1} more look(s) allowed.")})
    return {k: {"final": s["final"], "steps": s["steps"], "conv": s["conv"]} for k, s in state.items()}
