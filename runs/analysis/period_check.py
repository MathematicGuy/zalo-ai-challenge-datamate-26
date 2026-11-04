"""Do the samples use the month a question names? A check that needs no labels: the month is read from the question, the window from each sample's SQL. Aggregates only. python runs/analysis/period_check.py"""
import json, re, sys, statistics as st
from collections import Counter, defaultdict
from pathlib import Path
R = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(R / "sqlbase"), str(R / "src")]
import predict as P
from ingest import norm
from datamate.scoring import score_one
gold = {x["id"]: x for x in map(json.loads, open(R / "data/train/answers.jsonl"))}
rd = lambda p: [json.loads(x) for x in open(R / p, encoding="utf-8-sig") if x.strip()]
LONG = re.compile(r"(?i)(?:th[aá]ng|thangs?)\s*(\d{1,2})\s*(?:/|-|n[aă]m|nam)\s*(\d{4})")
SHORT = re.compile(r"(?i)(?<![\d/.\-])(?:(t|th|thg)\s*\.?\s*)?(\d{1,2})\s*[/\-.]\s*(\d{4}|\d{2})(?![\d/\-])")
def months(q):
    """[(style, year, month)] for the months a question names."""
    out = [("tháng M/Y", int(y), int(m)) for m, y in LONG.findall(q)]
    if out: return out
    for pre, m, y in SHORT.findall(q):
        y2 = int(y) if len(y) == 4 else 2000 + int(y)
        if 1 <= int(m) <= 12 and 2020 <= y2 <= 2030:
            out.append((("T/Th M/Y" if pre else "M/Y or M-Y") + (" two-digit year" if len(y) == 2 else ""), y2, int(m)))
    return out
def uses(sql, y, m):
    t = sql.lower()
    nxt = f"{y + (m == 12):04d}-{m % 12 + 1:02d}-01"; first = f"{y:04d}-{m:02d}-01"
    import calendar
    lastd = f"{y:04d}-{m:02d}-{calendar.monthrange(y, m)[1]:02d}"
    return first in t and (nxt in t or lastd in t) or re.search(rf"'{y:04d}-{m:02d}'", t) is not None or (re.search(rf"(month|extract\s*\(\s*month)[^)]*\)?\s*=\s*{m}\b", t) is not None and str(y) in t)
def run(name, trace, data, score=False):
    tr = {json.loads(l)["id"]: json.loads(l) for l in open(R / trace / "trace.jsonl", encoding="utf-8")}
    qs = {q["id"]: q for q in rd(f"{data}/questions.jsonl")}
    st_ = defaultdict(lambda: Counter()); ag = defaultdict(list); ex = defaultdict(list)
    for i, t in tr.items():
        ms = months(t["question"])
        if not ms: continue
        style = ms[0][0]
        for s in t["samples"][:8]:
            sql = P.parse(s["text"].split("\n#REPAIR\n")[-1])[1] or ""
            if not sql: st_[style]["no sql"] += 1; continue
            ok = all(uses(sql, y, m) for _, y, m in ms)
            st_[style]["uses it" if ok else "other window"] += 1
        hints = P.period_hints(t["question"], qs[i]["as_of_date"], None)
        st_[style]["questions"] += 1; st_[style]["with a hint"] += bool(hints)
        ss = t["samples"]; idx = P.debt_pool([x["text"] for x in ss], 8); a, c = P.vote([ss[j]["pred"] for j in idx], len(idx)); ag[style].append(c)
        st_[style]["abstain"] += not P.decide(a, c, .5)
        if score:
            r = a if P.decide(a, c, .5) else {"type": "abstain"}; st_[style][f"s{score_one({'id': i, **r}, gold[i])}"] += 1
    print(f"## {name}")
    for k, c in sorted(st_.items(), key=lambda kv: -kv[1]["questions"]):
        tot = c["uses it"] + c["other window"]
        print(f"  {k:30s} q={c['questions']:4d} hint {c['with a hint'] / c['questions']:4.0%} | samples using the month {c['uses it'] / max(1, tot):4.0%} (other {c['other window']}, no sql {c['no sql']}) | agree {st.mean(ag[k]):.2f} abstain {c['abstain'] / c['questions']:4.0%}" + (f" | right/wrong {c['s1']}/{c['s-1']}" if score else ""))
    print("  questions naming no month:", sum(not months(t['question']) for t in tr.values()), "of", len(tr))
run("public, LoRA v2", "runs/sql_v2_public", "data/public_test")
run("holdout, LoRA v2", "runs/sql_v2_holdout632", "data/train", True)
