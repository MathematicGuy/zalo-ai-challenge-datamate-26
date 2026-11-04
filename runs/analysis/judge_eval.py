"""A judge's P(Yes) on answer groups against the current vote, with thresholds chosen on the other shops. python runs/analysis/judge_eval.py JUDGE.jsonl"""
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
path = sys.argv[1]
J = {json.loads(l)["id"]: json.loads(l)["groups"] for l in open(R / path, encoding="utf-8")}
tr = {json.loads(l)["id"]: json.loads(l) for l in open(R / "runs/sql_v2_holdout632/trace.jsonl")}
S = lambda i, r: score_one({"id": i, **(r or {"type": "abstain"})}, gold[i])
def now(t):
    if t["final"].get("reason"): return t["final"]
    ss = t["samples"]; idx = P.debt_pool([s["text"] for s in ss], 8); ps = [ss[j]["pred"] for j in idx]
    a, c = P.vote(ps, len(ps)); return a if P.decide(a, c, .5) else {"type": "abstain"}
def auc(pos, neg):
    if not pos or not neg: return float("nan")
    x = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg]); rank = {}; i = 0
    while i < len(x):
        k = i
        while k < len(x) and x[k][0] == x[i][0]: k += 1
        rank[x[i][0]] = (i + k + 1) / 2; i = k
    return (sum(rank[v] for v in pos) - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
pos = [g["pyes"] for i, gs in J.items() for g in gs if S(i, g["pred"]) == 1]; neg = [g["pyes"] for i, gs in J.items() for g in gs if S(i, g["pred"]) == -1]
print(f"all answer groups: {len(pos)} right, {len(neg)} wrong, AUC {auc(pos, neg):.3f}; mean P(Yes) right {sum(pos)/max(1,len(pos)):.3f} wrong {sum(neg)/max(1,len(neg)):.3f}")
ans = [i for i in J if now(tr[i]).get("type") != "abstain" and not tr[i]["final"].get("reason")]
def group_of(i, r): return next((g for g in J[i] if P.answer_key(g["pred"]) == P.answer_key(r)), None)
pa_ = [group_of(i, now(tr[i]))["pyes"] for i in ans if S(i, now(tr[i])) == 1 and group_of(i, now(tr[i]))]; na_ = [group_of(i, now(tr[i]))["pyes"] for i in ans if S(i, now(tr[i])) == -1 and group_of(i, now(tr[i]))]
print(f"answers the current vote gives: {len(pa_)} right, {len(na_)} wrong, AUC {auc(pa_, na_):.3f}")
hi = [i for i in ans if P.vote([tr[i]["samples"][j]["pred"] for j in P.debt_pool([s["text"] for s in tr[i]["samples"]], 8)], len(P.debt_pool([s["text"] for s in tr[i]["samples"]], 8)))[1] >= .75]
pb = [group_of(i, now(tr[i]))["pyes"] for i in hi if S(i, now(tr[i])) == 1 and group_of(i, now(tr[i]))]; nb = [group_of(i, now(tr[i]))["pyes"] for i in hi if S(i, now(tr[i])) == -1 and group_of(i, now(tr[i]))]
print(f"answers at agreement >= .75: {len(pb)} right, {len(nb)} wrong, AUC {auc(pb, nb):.3f}")
shop = {i: tr[i]["business_id"] for i in ids}; shops = sorted(set(shop.values()))
def select(i, th):
    """votes x P(Yes) picks the group; it answers when share x P(Yes) reaches th."""
    if i not in J: return now(tr[i])
    best = max(J[i], key=lambda g: (g["share"] * g["pyes"], g["votes"]))
    return best["pred"] if best["share"] * best["pyes"] >= th and P.decide(best["pred"], 1.0, .5) else {"type": "abstain"}
def brake(i, th):
    r = now(tr[i])
    if i not in J or r.get("type") == "abstain": return r
    g = group_of(i, r)
    return {"type": "abstain"} if g and g["pyes"] < th else r
def rescue(i, th):
    r = now(tr[i])
    if i not in J or r.get("type") != "abstain": return r
    return select(i, th)
def both(i, th):
    r = brake(i, th[0])
    return rescue(i, th[1]) if now(tr[i]).get("type") == "abstain" else r
grid = [x / 20 for x in range(0, 20)]
def loso(fn, thetas):
    """Each shop uses the threshold that scores best on the other five."""
    out = {}
    for s in shops:
        tr_ids = [i for i in ids if shop[i] != s]; te = [i for i in ids if shop[i] == s]
        th = max(thetas, key=lambda t: sum(S(i, fn(i, t)) for i in tr_ids))
        out.update({i: S(i, fn(i, th)) for i in te})
    return out
base = {i: S(i, now(tr[i])) for i in ids}
cfg = {"current": base, "pick by votes x P(Yes)": loso(select, grid), "brake: abstain when P(Yes) is low": loso(brake, grid), "rescue: only on abstains": loso(rescue, grid),
       "brake + rescue": loso(both, [(a_, b_) for a_ in grid[::2] for b_ in grid[::2]])}
pa = lambda d, xs: (lambda v: (v.count(1) - v.count(-1)) / max(1, len(v)))([d[i] for i in xs])
wt = lambda d: sum(mix[k] / 1000 * pa(d, buckets[k]) for k in range(6))
print(f"{'config (thresholds from the other shops)':42s} {'all':>6s} {'near':>6s} {'mid':>6s} {'far':>6s} {'mix':>6s}   right/wrong/abstain")
for n, d in cfg.items():
    c = Counter(d.values()); print(f"{n:42s} {pa(d,ids):6.3f} {pa(d,near):6.3f} {pa(d,mid):6.3f} {pa(d,far):6.3f} {wt(d):6.3f}   {c[1]}/{c[-1]}/{c[0]}")
random.seed(0)
for n, d in list(cfg.items())[1:]:
    out = []
    for _ in range(2000):
        t = 0
        for k, xs in enumerate(buckets):
            s = random.choices(xs, k=len(xs)); t += mix[k] / 1000 * sum(d[i] - base[i] for i in s) / len(s)
        out.append(t)
    out.sort(); print(f"  mix delta, {n}: {wt(d)-wt(base):+.3f} CI95 [{out[50]:+.3f}, {out[1949]:+.3f}]")
