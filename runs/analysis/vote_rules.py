"""Offline vote rules on a run: stronger debt routing, answering unanswerable majorities. python runs/analysis/vote_rules.py RUN"""
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
tr = load(sys.argv[1]); MAC = re.compile(r"debt_(at|new|repaid)\s*\(")
DEBTQ = re.compile(r"(?:^|_)(no|cong_no|no_nan|ghi_no|mua_chiu|tra_no|thu_no|no_xau)(?:_|$)")
from ingest import norm
use = lambda s: bool(MAC.search(s["text"].split("\n#REPAIR\n")[-1]))
def decide(a, c, un):
    if un and a.get("type") == "unanswerable" and c >= .5: return a
    return a if P.decide(a, c, .5) else {"type": "abstain"}
def f(i, need=None, un=False, word=False):
    t = tr[i]
    if t["final"].get("reason"): return t["final"]
    ss = t["samples"]; idx = P.debt_pool([s["text"] for s in ss], 8)
    macro = [j for j, s in enumerate(ss) if use(s)]
    if need and len(macro) >= need and (not word or DEBTQ.search(norm(t["question"]))): idx = macro
    ps = [ss[j]["pred"] for j in idx]; a, c = P.vote(ps, len(ps)); return decide(a, c, un)
cfg = {"current rules": {}, "debt macro samples vote when 1+ use one": dict(need=1), "... when 2+ use one": dict(need=2), "... when 3+ use one": dict(need=3), "... 1+ and the question names debt": dict(need=1, word=True), "answer unanswerable majorities": dict(un=True), "2+ macro samples + unanswerable": dict(need=2, un=True)}
res = {n: {i: S(i, f(i, **k)) for i in ids} for n, k in cfg.items()}
pa = lambda d, xs: (lambda v: (v.count(1) - v.count(-1)) / max(1, len(v)))([d[i] for i in xs])
wt = lambda d: sum(mix[k] / 1000 * pa(d, buckets[k]) for k in range(6))
base = res["current rules"]; random.seed(0)
print(f"{'rule':42s} {'all':>6s} {'far':>6s} {'mix':>6s}  right/wrong/abstain  changed")
for n, d in res.items():
    c = Counter(d.values()); print(f"{n:42s} {pa(d,ids):6.3f} {pa(d,far):6.3f} {wt(d):6.3f}  {c[1]}/{c[-1]}/{c[0]}  {dict(Counter((base[i], d[i]) for i in ids if base[i] != d[i]))}")
