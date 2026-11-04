"""The reader and reviewer roles against the current setup: PA overall, near/mid/far, and reweighted to the public similarity mix. python runs/analysis/agents_eval.py"""
import json, math, re, sys, unicodedata
from collections import Counter, defaultdict
from pathlib import Path
R = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(R / "sqlbase"), str(R / "src")]
import predict as P, fewshot
from datamate.scoring import score_one
gold = {x["id"]: x for x in map(json.loads, open(R / "data/train/answers.jsonl"))}
rd = lambda p: [json.loads(l) for l in open(p, encoding="utf-8-sig") if l.strip()]
trq = {q["id"]: q for q in rd(R / "data/train/questions.jsonl")}; pub = rd(R / "data/public_test/questions.jsonl")
ids = [i.strip() for i in open(R / "splits/holdout600.txt") if i.strip()]
tr24 = [trq[i.strip()] for i in open(R / "splits/rft_train24.txt") if i.strip()]
F = [fewshot.feats(q["question"]) for q in tr24]; df = Counter(k for f in F for k in f); N = len(F)
idf = {k: math.log(1 + N / v) for k, v in df.items()}; nm = [math.sqrt(sum((c * idf[k]) ** 2 for k, c in f.items())) for f in F]
inv = defaultdict(list)
for j, f in enumerate(F):
    for k, c in f.items(): inv[k].append((j, c))
def nn(text):
    f = fewshot.feats(text); nq = math.sqrt(sum((c * idf.get(k, 1.0)) ** 2 for k, c in f.items())) or 1.0
    s = defaultdict(float)
    for k, c in f.items():
        for j, g in inv.get(k, ()): s[j] += c * g * idf[k] ** 2
    return max((v / (nq * nm[j]) for j, v in s.items()), default=0.0)
edges = [0, .5, .6, .7, .8, .9, 1.01]; bk = lambda x: next(k for k in range(6) if edges[k] <= x < edges[k + 1])
sim = {i: nn(trq[i]["question"]) for i in ids}; mix = Counter(bk(nn(q["question"])) for q in pub)
buckets = [[i for i in ids if bk(sim[i]) == k] for k in range(6)]
far = [i for i in ids if sim[i] < .7]; mid = [i for i in ids if .7 <= sim[i] < .9]; near = [i for i in ids if sim[i] >= .9]
tr = {json.loads(l)["id"]: json.loads(l) for l in open(R / "runs/sql_v2_holdout632/trace.jsonl")}

import random
load = lambda r: {json.loads(l)["id"]: json.loads(l) for l in open(R / r / "trace.jsonl")}
loadr = lambda p: {json.loads(l)["id"]: json.loads(l) for l in open(R / p)} if (R / p).exists() else {}
S = lambda i, r: score_one({"id": i, **(r or {"type": "abstain"})}, gold[i])
def now(t):
    if t["final"].get("reason"): return t["final"]
    ss = t["samples"]; idx = P.debt_pool([s["text"] for s in ss], 8); ps = [ss[j]["pred"] for j in idx]
    a, c = P.vote(ps, len(ps)); return a if P.decide(a, c, .5) else {"type": "abstain"}
def apply(t, rv, mode):
    """mode: rescue only changes abstains, brake only changes answers, both changes either; keep: NONE leaves the vote standing."""
    r = now(t)
    if not rv or not rv.get("choice") or t["final"].get("reason"): return r
    ab = r.get("type") == "abstain"
    if (mode.startswith("rescue") and not ab) or (mode.startswith("brake") and ab): return r
    if rv["choice"] == "NONE": return r if "keep" in mode else {"type": "abstain"}
    pick = next((c["pred"] for c in rv["candidates"] if c["label"] == rv["choice"]), None)
    return pick if pick and P.decide(pick, 1.0, .5) else {"type": "abstain"}
pa = lambda d, xs: (lambda v: (v.count(1) - v.count(-1)) / max(1, len(v)))([d[i] for i in xs])
def table(cfg, xs, base):
    bs = [[i for i in b if i in set(xs)] for b in buckets]
    wt = lambda d: sum(mix[k] / 1000 * pa(d, bs[k]) for k in range(6) if bs[k])
    sub = lambda g: [i for i in g if i in set(xs)]
    print(f"{'config':46s} {'all':>6s} {'near':>6s} {'mid':>6s} {'far':>6s} {'mix':>6s}   right/wrong/abstain  ({len(xs)} q)")
    for n, d in cfg.items():
        c = Counter(d[i] for i in xs); print(f"{n:46s} {pa(d,xs):6.3f} {pa(d,sub(near)):6.3f} {pa(d,sub(mid)):6.3f} {pa(d,sub(far)):6.3f} {wt(d):6.3f}   {c[1]}/{c[-1]}/{c[0]}")
    random.seed(0)
    for n, d in cfg.items():
        if d is base: continue
        out = []
        for _ in range(2000):
            t = 0
            for k, b in enumerate(bs):
                if b:
                    s = random.choices(b, k=len(b)); t += mix[k] / 1000 * sum(d[i] - base[i] for i in s) / len(s)
            out.append(t)
        out.sort(); print(f"  mix delta, {n}: {wt(d)-wt(base):+.3f} CI95 [{out[50]:+.3f}, {out[1949]:+.3f}]")
def review_cfg(tr, rv, xs, label):
    out = {f"{label}": {i: S(i, now(tr[i])) for i in xs}}
    for m, n in (("rescue", "reviewer on abstains only"), ("brake", "reviewer on answers only"), ("both", "reviewer on both, NONE abstains"), ("both keep", "reviewer on both, NONE keeps the vote")):
        out[f"{label} + {n}"] = {i: S(i, apply(tr[i], rv.get(i), m)) for i in xs}
    return out
def review_stats(rv, label):
    c = Counter()
    for i, x in rv.items():
        rs = [S(i, k["pred"]) == 1 for k in x["candidates"]]; has = any(rs); ch = x.get("choice")
        pick = next((k for k in x["candidates"] if k["label"] == ch), None)
        c[("right among candidates" if has else "no right candidate"), ("cut off" if ch is None else "NONE" if ch == "NONE" else "picked right" if pick and S(i, pick["pred"]) == 1 else "picked wrong")] += 1
    print(label, "reviewed", len(rv)); [print(f"  {a}: {b} = {v}") for (a, b), v in sorted(c.items())]
v2 = load("runs/sql_v2_holdout632"); rv = loadr("runs/review_v2_holdout632.jsonl")
if rv:
    review_stats(rv, "reviewer on LoRA v2:")
    cfg = review_cfg(v2, rv, ids, "current"); table(cfg, ids, cfg["current"])
if (R / "runs/sql_v2_reader350/trace.jsonl").exists():
    rd_ = load("runs/sql_v2_reader350"); rv2 = loadr("runs/review_v2_reader350.jsonl"); xs = [i for i in ids if i in rd_]
    print("reader: questions with rules", sum(bool(rd_[i].get("picks")) for i in xs), "of", len(xs), "| mean lines", round(sum(len(rd_[i].get("picks") or []) for i in xs) / max(1, len(xs)), 1))
    if rv2: review_stats(rv2, "reviewer on the reader run:")
    base = {i: S(i, now(v2[i])) for i in xs}; cfg = {"current": base}
    if rv: cfg["current + reviewer on both, NONE abstains"] = {i: S(i, apply(v2[i], rv.get(i), "both")) for i in xs}
    cfg.update(review_cfg(rd_, rv2, xs, "reader")); table(cfg, xs, base)
