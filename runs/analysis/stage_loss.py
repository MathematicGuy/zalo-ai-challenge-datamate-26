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
def pool(t):
    ss = t["samples"]; th = ss[8:]
    return [s for s in ss if use(s)] if th and sum(map(use, th)) / len(th) >= .5 else ss
S = lambda i, r: score_one({"id": i, **(r or {"type": "abstain"})}, gold[i])
def plain(a):
    return unicodedata.normalize("NFKD", str(a)).encode("ascii", "ignore").decode().lower().strip()
def canon(p):
    """Same answer written differently: number at 2 significant decimals, string without case and accents, list as a set, table as a set of rows of cells, column names and order ignored."""
    a = p.get("answer"); t = p["type"]
    if t in ("number", "percent") and isinstance(a, (int, float)) and not isinstance(a, bool): return ("num", round(float(a), 2))
    if t == "string": return ("str", plain(a))
    if t == "list" and isinstance(a, list): return ("list", tuple(sorted(plain(x) for x in a)))
    if t == "table" and isinstance(a, dict): return ("table", tuple(sorted(tuple(sorted(plain(c) for c in r)) for r in a.get("rows", []))))
    return (t, json.dumps(a, ensure_ascii=False, default=str, sort_keys=True))
def groups(ps, key):
    g = defaultdict(list)
    for p in ps:
        if p: g[key(p)].append(p)
    return sorted(g.values(), key=len, reverse=True)
def rep(g):
    return Counter(json.dumps(p, ensure_ascii=False, default=str, sort_keys=True) for p in g).most_common(1)[0][0]
def final_now(i):
    t = tr[i]
    if t["final"].get("reason"): return t["final"], 1.0
    ps = [s["pred"] for s in pool(t)]; a, c = P.vote(ps, len(ps))
    return (a if P.decide(a, c, .5) else {"type": "abstain"}), c
def final_canon(i):
    t = tr[i]
    if t["final"].get("reason"): return t["final"]
    ps = [s["pred"] for s in pool(t)]; gs = groups(ps, canon)
    if not gs: return {"type": "abstain"}
    a = json.loads(rep(gs[0])); c = len(gs[0]) / len(ps)
    return a if P.decide(a, c, .5) else {"type": "abstain"}
now = {i: final_now(i) for i in ids}; sc = {i: S(i, now[i][0]) for i in ids}; sc2 = {i: S(i, final_canon(i)) for i in ids}
pa = lambda d, xs: (lambda v: (v.count(1) - v.count(-1)) / max(1, len(v)))([d[i] for i in xs])
wt = lambda d: sum(mix[k] / 1000 * pa(d, buckets[k]) for k in range(6))
print(f"public mix: far {sum(mix[k] for k in range(3))/10:.1f}% mid {sum(mix[k] for k in (3,4))/10:.1f}% near {mix[5]/10:.1f}% | holdout far {len(far)} mid {len(mid)} near {len(near)}")
print(f"current: all {pa(sc,ids):.3f} near {pa(sc,near):.3f} mid {pa(sc,mid):.3f} far {pa(sc,far):.3f} public-mix {wt(sc):.3f}")
print(f"vote merging equivalent answers: all {pa(sc2,ids):.3f} near {pa(sc2,near):.3f} mid {pa(sc2,mid):.3f} far {pa(sc2,far):.3f} public-mix {wt(sc2):.3f}; rows changed {sum(sc[i]!=sc2[i] for i in ids)}")
# funnel
cat = {}
for i in ids:
    any_right = any(s["pred"] and S(i, s["pred"]) == 1 for s in tr[i]["samples"])
    cat[i] = ("1 right" if sc[i] == 1 else ("2 wrong, a right sample existed" if any_right else "3 wrong, no right sample") if sc[i] == -1 else ("4 abstain, a right sample existed" if any_right else "5 abstain, no right sample"))
print("funnel, share of questions: all | near | mid | far | public-mix")
for c in sorted(set(cat.values())):
    f = lambda xs: sum(cat[i] == c for i in xs) / len(xs)
    print(f"  {c:36s} {f(ids):6.1%} {f(near):6.1%} {f(mid):6.1%} {f(far):6.1%} {sum(mix[k]/1000*f(buckets[k]) for k in range(6)):6.1%}")
orc = {i: 1 if "right" in cat[i] and "no right" not in cat[i] else 0 for i in ids}
print(f"perfect selector among current samples (right when any sample is, else abstain): all {pa(orc,ids):.3f} far {pa(orc,far):.3f} public-mix {wt(orc):.3f}")
# a judge on the top two groups, for questions whose top group holds under 75%
for p_ in (.6, .7, .8, .9):
    ex = {}
    for i in ids:
        t = tr[i]; ex[i] = sc[i]
        if t["final"].get("reason"): continue
        ps = [s["pred"] for s in pool(t)]; gs = groups(ps, canon)
        if len(gs) < 2 or len(gs[0]) / max(1, len(ps)) >= .75 or len(gs[1]) < 2: continue
        rs = [S(i, json.loads(rep(g))) == 1 for g in gs[:2]]
        ex[i] = (2 * p_ - 1) if any(rs) else -1
    print(f"judge right {p_:.0%} of the time when the top two hold the answer: all {pa_ if False else sum(ex.values())/len(ids):.3f} far {sum(ex[i] for i in far)/len(far):.3f} public-mix {sum(mix[k]/1000*sum(ex[i] for i in buckets[k])/len(buckets[k]) for k in range(6)):.3f}")
# how wrong are the wrong answers
kinds = Counter(); agr = Counter(); lay = Counter(); byt = defaultdict(Counter)
for i in ids:
    g = gold[i]; byt[g["type"]][sc[i]] += 1
    if sc[i] != -1: continue
    r, c = now[i]; agr["0.9+" if c >= .9 else "0.75-0.9" if c >= .75 else "<0.75"] += 1
    a, b = r.get("answer"), g.get("answer")
    if r["type"] != g["type"]: kinds["wrong type"] += 1
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and b:
        q = abs(a - b) / abs(b); kinds["number within 5%" if q < .05 else "number within 25%" if q < .25 else "number off by more"] += 1
    elif isinstance(a, list) and isinstance(b, list):
        x, y = {plain(v) for v in a}, {plain(v) for v in b}; j = len(x & y) / max(1, len(x | y)); kinds["list overlap >=.5" if j >= .5 else "list overlap <.5"] += 1
    else: kinds[f"other {g['type']}"] += 1
    tb = [fewshot.tables_of(s["text"].split("\n#REPAIR\n")[-1].split("</think>")[-1]) for s in pool(tr[i]) if s["pred"]]
    can = sum(bool(x) and x <= fewshot.CANON for x in tb) / max(1, len(tb)); lay["mostly views/macros" if can >= .5 else "mostly raw tables"] += 1
print("wrong answers", sum(v == -1 for v in sc.values()), "kind", kinds.most_common(), "| agreement", agr.most_common(), "| source", lay.most_common())
print("gold type: n right/wrong/abstain:", {k: (sum(v.values()), v[1], v[-1], v[0]) for k, v in sorted(byt.items())})
