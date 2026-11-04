"""The exploration agent on hard questions against 4 thinking samples. python runs/analysis/explore_eval.py AGENT.jsonl"""
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
A = {json.loads(l)["id"]: json.loads(l)["samples"] for l in open(R / sys.argv[1], encoding="utf-8")}
tr = {json.loads(l)["id"]: json.loads(l) for l in open(R / "runs/sql_v2_holdout632/trace.jsonl")}
S = lambda i, r: score_one({"id": i, **(r or {"type": "abstain"})}, gold[i])
def vote(ss, start=None):
    idx = P.debt_pool([s["text"] for s in ss], start) if start is not None else range(len(ss))
    ps = [ss[j]["pred"] for j in idx]; a, c = P.vote(ps, len(ps)); return a if P.decide(a, c, .5) else {"type": "abstain"}
def f(i, how):
    t = tr[i]
    if t["final"].get("reason"): return t["final"]
    ss = t["samples"]
    if i not in A or how == "current": return vote(ss, 8)
    if how == "mix": return vote(ss[:8] + A[i], 8)
    return vote(A[i], 0)
cfg = {n: {i: S(i, f(i, k)) for i in ids} for n, k in (("current: 8 plain + 4 thinking on hard", "current"), ("8 plain + 4 agent runs on hard", "mix"), ("hard decided by 4 agent runs only", "alone"))}
base = cfg["current: 8 plain + 4 thinking on hard"]; hard = [i for i in ids if i in A]
pa = lambda d, xs: (lambda v: (v.count(1) - v.count(-1)) / max(1, len(v)))([d[i] for i in xs])
wt = lambda d: sum(mix[k] / 1000 * pa(d, buckets[k]) for k in range(6))
print(f"hard questions {len(hard)} of {len(ids)}; agent steps per run {sum(len(s['steps']) for i in hard for s in A[i]) / max(1, 4 * len(hard)):.2f}; kinds {Counter(x for i in hard for s in A[i] for x in s['steps']).most_common()}")
print(f"single samples right: thinking {sum(bool(s['pred']) and S(i, s['pred']) == 1 for i in hard for s in tr[i]['samples'][8:])}, agent {sum(bool(s['pred']) and S(i, s['pred']) == 1 for i in hard for s in A[i])}, of {4 * len(hard)}")
print(f"{'config':42s} {'all':>6s} {'hard':>6s} {'near':>6s} {'mid':>6s} {'far':>6s} {'mix':>6s}   right/wrong/abstain")
for n, d in cfg.items():
    c = Counter(d.values()); print(f"{n:42s} {pa(d,ids):6.3f} {pa(d,hard):6.3f} {pa(d,near):6.3f} {pa(d,mid):6.3f} {pa(d,far):6.3f} {wt(d):6.3f}   {c[1]}/{c[-1]}/{c[0]}")
random.seed(0)
for n, d in list(cfg.items())[1:]:
    out = []
    for _ in range(2000):
        t = 0
        for k, xs in enumerate(buckets):
            s = random.choices(xs, k=len(xs)); t += mix[k] / 1000 * sum(d[i] - base[i] for i in s) / len(s)
        out.append(t)
    out.sort(); print(f"  mix delta, {n}: {wt(d)-wt(base):+.3f} CI95 [{out[50]:+.3f}, {out[1949]:+.3f}]")
