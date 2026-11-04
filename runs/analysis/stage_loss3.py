"""Per-shop ingest signals beside agreement, and views against raw tables inside the same question. Counts only. python runs/analysis/stage_loss3.py"""
import json, re, sys, statistics as st
from collections import Counter, defaultdict
from pathlib import Path
R = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(R / "sqlbase"), str(R / "src")]
import fewshot
from ingest import load_business
from datamate.scoring import score_one
rd = lambda p: [json.loads(l) for l in open(p, encoding="utf-8-sig") if l.strip()]
gold = {x["id"]: x for x in map(json.loads, open(R / "data/train/answers.jsonl"))}
trp = {json.loads(l)["id"]: json.loads(l) for l in open(R / "runs/sql_v2_public/trace.jsonl")}
tr = {json.loads(l)["id"]: json.loads(l) for l in open(R / "runs/sql_v2_holdout632/trace.jsonl")}
by = defaultdict(list)
for t in trp.values(): by[t["business_id"]].append(t)
print("public shop: tables rows | numeric-looking text columns: with separators or units, digits only | half-empty columns | answered, agreement")
for b in sorted(by):
    db, meta = load_business(Path("data/public_test"), b)
    names = [r[0] for r in db.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='main'").fetchall()]
    sep = dig = he = cols = rows = 0
    for n in names:
        cnt = db.execute(f'SELECT COUNT(*) FROM "{n}"').fetchone()[0]; rows += cnt
        for c_ in db.execute(f'DESCRIBE "{n}"').fetchall():
            col, typ = c_[0], c_[1]; cols += 1
            if col.startswith("_") or not cnt: continue
            nn = db.execute(f'SELECT COUNT("{col}") FROM "{n}"').fetchone()[0]; he += nn < cnt * .5
            if typ != "VARCHAR" or not nn: continue
            d, s = db.execute(f"""SELECT COUNT(*) FILTER (WHERE regexp_full_match(trim("{col}"), '[0-9]+')), COUNT(*) FILTER (WHERE regexp_full_match(trim("{col}"), '[-+]?[0-9][0-9., ]*[%đ₫dDkKmM]*')) FROM "{n}" WHERE "{col}" IS NOT NULL""").fetchone()
            if s >= .9 * nn: (dig if d >= .9 * nn else 0); dig += d >= .9 * nn; sep += d < .9 * nn
    ts = by[b]; print(f"  {b}: {len(names)} tables, {rows} rows, {cols} cols | {sep}, {dig} | {he} | {sum(t['final'].get('type') != 'abstain' for t in ts)/len(ts):.2f}, {st.mean(t['conf'] for t in ts):.2f}")
    db.close()
kind = lambda s: (lambda x: "views" if x and x <= fewshot.CANON else "raw" if x and not (x & fewshot.CANON) else "both" if x else "none")(fewshot.tables_of(s["text"].split("\n#REPAIR\n")[-1].split("</think>")[-1]))
c = Counter(); qs = Counter()
for i, t in tr.items():
    per = defaultdict(list)
    for s in t["samples"][:8]:
        if s["pred"]: per[kind(s)].append(score_one({"id": i, **s["pred"]}, gold[i]) == 1)
    if per["views"] and per["raw"]:
        qs["questions with both"] += 1
        for k in ("views", "raw"): c[k, "right"] += sum(per[k]); c[k, "n"] += len(per[k])
        m = lambda k: sum(per[k]) / len(per[k])
        qs["views better" if m("views") > m("raw") else "raw better" if m("raw") > m("views") else "tie"] += 1
    for k, v in per.items(): c["all", k] += len(v)
print("plain samples inside the same question:", dict(qs), {k: f"{c[k,'right']}/{c[k,'n']}" for k in ("views", "raw")}, "| source of all plain samples", {k: c["all", k] for k in ("views", "raw", "both", "none")})
