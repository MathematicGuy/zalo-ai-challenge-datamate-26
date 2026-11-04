"""Per-shop signals against quality: are weak shops the ones whose columns nothing explains? Aggregates only. python runs/analysis/shop_factors.py"""
import json, math, re, statistics as st, sys
from collections import Counter, defaultdict
from pathlib import Path
R = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(R / "sqlbase"), str(R / "src")]
import predict as P
from ingest import load_business, norm
from datamate.scoring import score_one
gold = {x["id"]: x for x in map(json.loads, open(R / "data/train/answers.jsonl"))}
load = lambda r: {json.loads(l)["id"]: json.loads(l) for l in open(R / r / "trace.jsonl")}
H, Q = load("runs/sql_v2_holdout632"), load("runs/sql_v2_public")
def final(t):
    if t["final"].get("reason"): return t["final"], 1.0
    ss = t["samples"]; idx = P.debt_pool([s["text"] for s in ss], 8); ps = [ss[j]["pred"] for j in idx]; a, c = P.vote(ps, len(ps))
    return (a if P.decide(a, c, .5) else {"type": "abstain"}), c
def doc_cover(root, b):
    db, meta = load_business(Path(root), b)
    rd = norm(meta["readme"]); names = [r[0] for r in db.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='main'").fetchall()]
    cols = hit = 0; opaque = 0
    for n in names:
        for c_ in db.execute(f'DESCRIBE "{n}"').fetchall():
            col = c_[0]
            if col.startswith("_"): continue
            k = norm(col); cols += 1; hit += bool(k) and k in rd
            opaque += len(k) <= 3 or bool(re.fullmatch(r"(col|cot|field|f|c)_?\d+", k))
    db.close()
    return dict(tables=len(names), cols=cols, doc=hit / max(1, cols), opaque=opaque / max(1, cols), readme=len(meta["readme"]))
rows = []
for name, tr, root in (("holdout", H, "data/train"), ("public", Q, "data/public_test")):
    by = defaultdict(list)
    for i, t in tr.items(): by[t["business_id"]].append((i, t))
    for b, xs in sorted(by.items()):
        f = doc_cover(root, b); fs = [final(t) for _, t in xs]
        r = dict(split=name, shop=b, n=len(xs), agree=st.mean(c for _, c in fs), answered=sum(a["type"] != "abstain" for a, _ in fs) / len(xs), **f)
        if name == "holdout":
            s = [score_one({"id": i, **a}, gold[i]) for (i, _), (a, _) in zip(xs, fs)]; r["pa"] = (s.count(1) - s.count(-1)) / len(s)
        rows.append(r)
print("shop n tables cols | doc% opaque% readme | agree answered PA")
for r in rows:
    print(f"{r['shop']:8s} {r['n']:4d} {r['tables']:3d} {r['cols']:4d} | {r['doc']:4.0%} {r['opaque']:4.0%} {r['readme']:6d} | {r['agree']:.2f} {r['answered']:.2f} {r.get('pa', float('nan')):+.2f}")
def corr(x, y):
    mx, my = st.mean(x), st.mean(y); a = sum((p - mx) * (q - my) for p, q in zip(x, y)); b = math.sqrt(sum((p - mx) ** 2 for p in x) * sum((q - my) ** 2 for q in y)); return a / b if b else float("nan")
for k in ("doc", "opaque", "readme", "tables", "cols"):
    print(f"corr(agreement, {k}) over {len(rows)} shops: {corr([r[k] for r in rows], [r['agree'] for r in rows]):+.2f}")
# unanswerable top groups, offline
c = Counter(); mix = Counter()
for i, t in H.items():
    if t["final"].get("reason"): continue
    ss = t["samples"]; idx = P.debt_pool([s["text"] for s in ss], 8); a, cf = P.vote([ss[j]["pred"] for j in idx], len(idx))
    if a.get("type") == "unanswerable" and cf >= .5: c["unanswerable top, gold unanswerable" if gold[i]["type"] == "unanswerable" else "unanswerable top, gold has an answer"] += 1
print(dict(c), "| public unanswerable top groups:", sum(1 for t in Q.values() if not t["final"].get("reason") and (lambda ss, idx: (lambda r: r[0].get("type") == "unanswerable" and r[1] >= .5)(P.vote([ss[j]["pred"] for j in idx], len(idx))))(t["samples"], P.debt_pool([s["text"] for s in t["samples"]], 8))))
# how close are right answers to the edge of the local scorer
near = Counter()
for i, t in H.items():
    a, _ = final(t); g = gold[i]
    if a.get("type") in ("number", "percent") and g["type"] == a["type"] and score_one({"id": i, **a}, g) == 1:
        x, y = float(a["answer"]), float(g["answer"]); near["exact" if x == y else "within 0.01" if abs(x - y) <= .01 else "within 0.5%" if abs(x - y) <= .005 * max(1, abs(y)) else "wider"] += 1
    elif a.get("type") == g["type"] and score_one({"id": i, **a}, g) == 1: near[f"{g['type']} right"] += 1
print("right answers vs gold:", near.most_common())
