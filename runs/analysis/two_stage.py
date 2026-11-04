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

MAC = re.compile(r"debt_(at|new|repaid)\s*\(")
use = lambda s: bool(MAC.search(s["text"].split("\n#REPAIR\n")[-1]))
load = lambda r: {json.loads(l)["id"]: json.loads(l) for l in open(R / r / "trace.jsonl")}
v2, to = load("runs/sql_v2_holdout632"), load("runs/sql_v2_thinkonly632")
S = lambda i, r: score_one({"id": i, **(r or {"type": "abstain"})}, gold[i])
def vote(ss, th=.5, route=False):
    if route and ss and sum(map(use, ss)) / len(ss) >= .5: ss = [s for s in ss if use(s)]
    ps = [s["pred"] for s in ss]; a, c = P.vote(ps, len(ps)); return a if P.decide(a, c, th) else {"type": "abstain"}
def cur(i):
    t = v2[i]
    if t["final"].get("reason"): return t["final"]
    ss = t["samples"]; th = ss[8:]
    if th and sum(map(use, th)) / len(th) >= .5: ss = [s for s in ss if use(s)]
    ps = [s["pred"] for s in ss]; a, c = P.vote(ps, len(ps)); return a if P.decide(a, c, .5) else {"type": "abstain"}
def stage(i, src, k, th=.5):
    """Easy questions (plain share >= .75) keep the plain vote; a hard question is decided by its thinking samples alone."""
    t = v2[i]
    if t["final"].get("reason"): return t["final"]
    ss = t["samples"]
    if len(ss) <= 8: return vote(ss)
    tt = to[i]["samples"][2:2 + k] if src == "new" else ss[8:8 + k]
    return vote(tt, th, route=True)
pa = lambda d, xs: (lambda v: (v.count(1) - v.count(-1)) / max(1, len(v)))([d[i] for i in xs])
wt = lambda d: sum(mix[k] / 1000 * pa(d, buckets[k]) for k in range(6))
base = {i: S(i, cur(i)) for i in ids}
cfg = {"current (8 plain + 4 thinking mixed on hard)": base}
for src, k in (("old", 4), ("new", 4), ("new", 6), ("new", 8)):
    cfg[f"hard decided by thinking only, K={k} ({'same run' if src == 'old' else 'think-only run'})"] = {i: S(i, stage(i, src, k)) for i in ids}
cfg["hard: K=8 thinking, threshold .4"] = {i: S(i, stage(i, "new", 8, .4)) for i in ids}
import random
random.seed(0)
print(f"{'config':58s} {'all':>6s} {'near':>6s} {'mid':>6s} {'far':>6s} {'mix':>6s}   right/wrong/abstain")
for n, d in cfg.items():
    c = Counter(d.values()); print(f"{n:58s} {pa(d,ids):6.3f} {pa(d,near):6.3f} {pa(d,mid):6.3f} {pa(d,far):6.3f} {wt(d):6.3f}   {c[1]}/{c[-1]}/{c[0]}")
for n, d in list(cfg.items())[1:]:
    out = []
    for _ in range(2000):
        t = 0
        for k, xs in enumerate(buckets):
            s = random.choices(xs, k=len(xs)); t += mix[k] / 1000 * sum(d[i] - base[i] for i in s) / len(s)
        out.append(t)
    out.sort(); print(f"mix delta vs current, {n}: {wt(d)-wt(base):+.3f} CI95 [{out[50]:+.3f}, {out[1949]:+.3f}]")
print("hard questions", sum(len(v2[i]["samples"]) > 8 for i in ids), "of", len(ids))
