"""Where the holdout loses points, by how close each question is to train, reweighted to the public similarity mix (diagnostic only: public questions never feed training).
python runs/analysis/stage_loss.py"""
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
last = lambda s: s["text"].split("\n#REPAIR\n")[-1]
use = lambda s: bool(MAC.search(last(s)))
tbl = lambda s: fewshot.tables_of(last(s).split("</think>")[-1])
kind = lambda s: (lambda x: "views" if x and x <= fewshot.CANON else "raw" if x and not (x & fewshot.CANON) else "both" if x else "none")(tbl(s))
load = lambda r: {json.loads(l)["id"]: json.loads(l) for l in open(R / r / "trace.jsonl")}
v2, to = load("runs/sql_v2_holdout632"), load("runs/sql_v2_thinkonly632")
S = lambda i, r: score_one({"id": i, **(r or {"type": "abstain"})}, gold[i])
def debt_pool(ss, start):
    th = ss[start:]
    return [s for s in ss if use(s)] if th and sum(map(use, th)) / len(th) >= .5 else ss
def decide(ss, th=.5):
    ps = [s["pred"] for s in ss]; a, c = P.vote(ps, len(ps)); return a if P.decide(a, c, th) else {"type": "abstain"}
def route(i, t, start, keep, need):
    if t["final"].get("reason"): return t["final"]
    ss = debt_pool(t["samples"][start:] if start else t["samples"], 0 if start else 8)
    ok = [s for s in ss if s["pred"] and kind(s) in keep]
    return decide(ok if len(ok) >= need else ss)
pa = lambda d, xs: (lambda v: (v.count(1) - v.count(-1)) / max(1, len(v)))([d[i] for i in xs])
wt = lambda d: sum(mix[k] / 1000 * pa(d, buckets[k]) for k in range(6))
base = {i: S(i, route(i, v2[i], 0, set(), 10**6)) for i in ids}
cfg = {"current": base}
for need in (2, 3, 4):
    cfg[f"views-only samples vote when at least {need}"] = {i: S(i, route(i, v2[i], 0, {"views"}, need)) for i in ids}
    cfg[f"no raw-only samples when at least {need} remain"] = {i: S(i, route(i, v2[i], 0, {"views", "both"}, need)) for i in ids}
cfg["think-only K=6 (reference)"] = {i: S(i, route(i, to[i], 2, set(), 10**6) if False else (lambda t: t["final"] if t["final"].get("reason") else decide(debt_pool(t["samples"][2:8], 0)))(to[i])) for i in ids}
cfg["think-only K=6, views-only when at least 2"] = {}
for i in ids:
    t = to[i]
    if t["final"].get("reason"): cfg["think-only K=6, views-only when at least 2"][i] = S(i, t["final"]); continue
    ss = debt_pool(t["samples"][2:8], 0); ok = [s for s in ss if s["pred"] and kind(s) == "views"]
    cfg["think-only K=6, views-only when at least 2"][i] = S(i, decide(ok if len(ok) >= 2 else ss))
print(f"{'config':46s} {'all':>6s} {'near':>6s} {'mid':>6s} {'far':>6s} {'mix':>6s}   right/wrong/abstain")
for n, d in cfg.items():
    c = Counter(d.values()); print(f"{n:46s} {pa(d,ids):6.3f} {pa(d,near):6.3f} {pa(d,mid):6.3f} {pa(d,far):6.3f} {wt(d):6.3f}   {c[1]}/{c[-1]}/{c[0]}")
random.seed(0)
for n in list(cfg)[1:]:
    d = cfg[n]; out = []
    for _ in range(2000):
        t = 0
        for k, xs in enumerate(buckets):
            s = random.choices(xs, k=len(xs)); t += mix[k] / 1000 * sum(d[i] - base[i] for i in s) / len(s)
        out.append(t)
    out.sort(); print(f"mix delta, {n}: {wt(d)-wt(base):+.3f} CI95 [{out[50]:+.3f}, {out[1949]:+.3f}]")
# public: how often the rule would change the vote (no labels)
tp = load("runs/sql_v2_public"); ch = Counter()
for i, t in tp.items():
    a = t["final"] if t["final"].get("reason") else decide(debt_pool(t["samples"], 8))
    ss = debt_pool(t["samples"], 8); ok = [s for s in ss if s["pred"] and kind(s) == "views"]
    b = t["final"] if t["final"].get("reason") else decide(ok if len(ok) >= 3 else ss)
    ch["same" if a == b else "abstain->answer" if a.get("type") == "abstain" else "answer->abstain" if b.get("type") == "abstain" else "changed answer"] += 1
print("public rows under views-only at least 3:", dict(ch))
