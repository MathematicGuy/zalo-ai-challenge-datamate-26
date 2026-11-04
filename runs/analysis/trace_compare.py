"""Compares runs on the holdout: PA overall, near/mid/far, reweighted to the public similarity mix. python runs/analysis/trace_compare.py BASE_RUN RUN [RUN ...]"""
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
S = lambda i, r: score_one({"id": i, **(r or {"type": "abstain"})}, gold[i])
def now(t):
    if t["final"].get("reason"): return t["final"]
    ss = t["samples"]; idx = P.debt_pool([s["text"] for s in ss], 8); ps = [ss[j]["pred"] for j in idx]
    a, c = P.vote(ps, len(ps)); return a if P.decide(a, c, .5) else {"type": "abstain"}
pa = lambda d, xs: (lambda v: (v.count(1) - v.count(-1)) / max(1, len(v)))([d[i] for i in xs])
runs = sys.argv[1:]; res = {r: {i: S(i, now(t)) for i, t in load(r).items() if i in set(ids)} for r in runs}
xs = [i for i in ids if all(i in res[r] for r in runs)]; bs = [[i for i in b if i in set(xs)] for b in buckets]
wt = lambda d: sum(mix[k] / 1000 * pa(d, bs[k]) for k in range(6) if bs[k])
sub = lambda g: [i for i in g if i in set(xs)]
print(f"{'run':30s} {'all':>6s} {'near':>6s} {'mid':>6s} {'far':>6s} {'mix':>6s}   right/wrong/abstain  ({len(xs)} q)")
for r in runs:
    d = res[r]; c = Counter(d[i] for i in xs); print(f"{r[-30:]:30s} {pa(d,xs):6.3f} {pa(d,sub(near)):6.3f} {pa(d,sub(mid)):6.3f} {pa(d,sub(far)):6.3f} {wt(d):6.3f}   {c[1]}/{c[-1]}/{c[0]}")
random.seed(0); base = res[runs[0]]
for r in runs[1:]:
    d = res[r]; out = []
    for _ in range(2000):
        t = 0
        for k, b in enumerate(bs):
            if b:
                s = random.choices(b, k=len(b)); t += mix[k] / 1000 * sum(d[i] - base[i] for i in s) / len(s)
        out.append(t)
    out.sort(); print(f"  mix delta, {r}: {wt(d)-wt(base):+.3f} CI95 [{out[50]:+.3f}, {out[1949]:+.3f}]")
    print("  changed rows:", dict(Counter((base[i], d[i]) for i in xs if base[i] != d[i])))
    shops = defaultdict(lambda: [Counter(), Counter()])
    tr = load(r)
    for i in xs:
        shops[tr[i]["business_id"]][0][base[i]] += 1; shops[tr[i]["business_id"]][1][d[i]] += 1
    print("  by shop PA:", {b: (round((a[1]-a[-1])/sum(a.values()), 2), round((c_[1]-c_[-1])/sum(c_.values()), 2)) for b, (a, c_) in sorted(shops.items())})
