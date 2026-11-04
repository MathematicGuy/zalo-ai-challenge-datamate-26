"""Qwen3.5-4B next to the 9B + LoRA v2 trace on the holdout: alone, pooled, as a rescue for 9B abstains, as a brake on 9B answers. python runs/analysis/q4b_pool.py"""
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
MAC = re.compile(r"debt_(at|new|repaid)\s*\(")
use = lambda s: bool(MAC.search(s["text"].split("\n#REPAIR\n")[-1]))
load = lambda r: {json.loads(l)["id"]: json.loads(l) for l in open(R / r / "trace.jsonl")}
v2, q4 = load("runs/sql_v2_holdout632"), load("runs/sql_q4b_holdout632")
S = lambda i, r: score_one({"id": i, **(r or {"type": "abstain"})}, gold[i])
def pool(t):
    ss = t["samples"]; th = ss[8:]
    return [s for s in ss if use(s)] if th and sum(map(use, th)) / len(th) >= .5 else ss
def top(ss):
    ps = [s["pred"] for s in ss]; a, c = P.vote(ps, max(1, len(ps))); return a, c
def same(a, b):
    k = lambda p: ("num", round(float(p["answer"]), 2)) if p.get("type") in ("number", "percent") and isinstance(p.get("answer"), (int, float)) and not isinstance(p.get("answer"), bool) else json.dumps(p, ensure_ascii=False, default=str, sort_keys=True)
    return a.get("type") != "abstain" and b.get("type") != "abstain" and k(a) == k(b)
def on(a, c, th=.5): return a if P.decide(a, c, th) else {"type": "abstain"}
def reasoned(t): return t["final"] if t["final"].get("reason") else None
def f_9(i):
    a, c = top(pool(v2[i])); return reasoned(v2[i]) or on(a, c)
def f_4(i):
    a, c = top(pool(q4[i])); return reasoned(q4[i]) or on(a, c)
def f_pool(i):
    a, c = top(pool(v2[i]) + pool(q4[i])); return reasoned(v2[i]) or on(a, c)
def f_rescue(i, need=.25):
    r = f_9(i)
    if r.get("type") != "abstain" or reasoned(v2[i]): return r
    a, c = top(pool(v2[i])); b, d = top(pool(q4[i]))
    return a if same(a, b) and c >= need and P.decide(a, 1.0, .5) else r
def f_brake(i, need=.5):
    r = f_9(i)
    if r.get("type") == "abstain" or reasoned(v2[i]): return r
    b, d = top(pool(q4[i]))
    return {"type": "abstain"} if d >= need and P.decide(b, d, need) and not same(r, b) else r
def f_both(i):
    r = f_brake(i)
    return f_rescue(i) if r.get("type") == "abstain" and f_9(i).get("type") == "abstain" else r
cfg = {"9B + LoRA v2 (current)": f_9, "4B alone": f_4, "pool all samples": f_pool, "rescue: 9B abstains, 4B agrees with its top group": f_rescue,
       "brake: 4B majority (>=.5) disagrees -> abstain": f_brake, "brake only when 4B majority >=.75": lambda i: f_brake(i, .75), "rescue + brake": f_both}
res = {n: {i: S(i, f(i)) for i in ids} for n, f in cfg.items()}
pa = lambda d, xs: (lambda v: (v.count(1) - v.count(-1)) / max(1, len(v)))([d[i] for i in xs])
wt = lambda d: sum(mix[k] / 1000 * pa(d, buckets[k]) for k in range(6))
print(f"{'config':52s} {'all':>6s} {'near':>6s} {'mid':>6s} {'far':>6s} {'mix':>6s}   right/wrong/abstain")
for n, d in res.items():
    c = Counter(d.values()); print(f"{n:52s} {pa(d,ids):6.3f} {pa(d,near):6.3f} {pa(d,mid):6.3f} {pa(d,far):6.3f} {wt(d):6.3f}   {c[1]}/{c[-1]}/{c[0]}")
base = res["9B + LoRA v2 (current)"]; random.seed(0)
for n, d in list(res.items())[1:]:
    out = []
    for _ in range(2000):
        t = 0
        for k, xs in enumerate(buckets):
            s = random.choices(xs, k=len(xs)); t += mix[k] / 1000 * sum(d[i] - base[i] for i in s) / len(s)
        out.append(t)
    out.sort(); print(f"mix delta, {n}: {wt(d)-wt(base):+.3f} CI95 [{out[50]:+.3f}, {out[1949]:+.3f}]")
# are the two models' mistakes independent?
c = Counter()
for i in ids:
    a, _ = top(pool(v2[i])); b, _ = top(pool(q4[i])); r9 = S(i, f_9(i))
    if r9 == -1: c["9B wrong, 4B same answer" if same(a, b) else "9B wrong, 4B differs"] += 1
    if r9 == 1: c["9B right, 4B same answer" if same(a, b) else "9B right, 4B differs"] += 1
    any9 = any(s["pred"] and S(i, s["pred"]) == 1 for s in v2[i]["samples"]); any4 = any(s["pred"] and S(i, s["pred"]) == 1 for s in q4[i]["samples"])
    if not any9: c["no right 9B sample: 4B has one" if any4 else "no right 9B sample: 4B has none"] += 1
print(dict(sorted(c.items())))
g = sum(len(t["samples"]) > 8 for t in q4.values()); print("4B hard questions", g, "of", len(q4), "| right samples 9B/4B", *(f"{sum(bool(s['pred']) and S(i, s['pred']) == 1 for s in tr_[i]['samples'][:8])/ (8*len(ids)):.3f}" for tr_ in (v2, q4)))
