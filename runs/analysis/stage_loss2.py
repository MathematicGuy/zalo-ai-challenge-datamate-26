"""Abstain detail, context sizes, ingest quality and public-only rule terms; prints aggregates only. python runs/analysis/stage_loss2.py"""
import json, math, re, sys, glob, os, statistics as st
from collections import Counter, defaultdict
from pathlib import Path
R = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(R / "sqlbase"), str(R / "src"), str(R / "runs/analysis")]
import predict as P, fewshot, llmviews, debtview
from ingest import load_business, schema_text, norm
from datamate.scoring import score_one
from tokenizers import Tokenizer
tok = Tokenizer.from_file(str(R / "sqlbase/models/Qwen3.5-9B/tokenizer.json")); nt = lambda s: len(tok.encode(s or "").ids)
rd = lambda p: [json.loads(l) for l in open(p, encoding="utf-8-sig") if l.strip()]
gold = {x["id"]: x for x in map(json.loads, open(R / "data/train/answers.jsonl"))}
trq = {q["id"]: q for q in rd(R / "data/train/questions.jsonl")}; pub = rd(R / "data/public_test/questions.jsonl")
ids = [i.strip() for i in open(R / "splits/holdout600.txt") if i.strip()]
tr = {json.loads(l)["id"]: json.loads(l) for l in open(R / "runs/sql_v2_holdout632/trace.jsonl")}
trp = {json.loads(l)["id"]: json.loads(l) for l in open(R / "runs/sql_v2_public/trace.jsonl")}
MAC = re.compile(r"debt_(at|new|repaid)\s*\(")
use = lambda s: bool(MAC.search(s["text"].split("\n#REPAIR\n")[-1]))
def pool(t):
    ss = t["samples"]; th = ss[8:]
    return [s for s in ss if use(s)] if th and sum(map(use, th)) / len(th) >= .5 else ss
S = lambda i, r: score_one({"id": i, **(r or {"type": "abstain"})}, gold[i])
dump = lambda p: json.dumps(p, ensure_ascii=False, default=str, sort_keys=True)
def key(p):
    a = p.get("answer")
    return dump(["num", round(float(a), 2)]) if p["type"] in ("number", "percent") and isinstance(a, (int, float)) and not isinstance(a, bool) else dump(p)
def gs(t):
    g = defaultdict(list)
    for s in pool(t):
        if s["pred"]: g[key(s["pred"])].append(s["pred"])
    return sorted(g.values(), key=len, reverse=True), len(pool(t))
# 1. abstained questions: where does the right answer sit
rank = Counter(); ex = defaultdict(Counter); nil = Counter()
for i in ids:
    t = tr[i]
    if t["final"].get("reason"): continue
    g, n = gs(t)
    a, c = P.vote([s["pred"] for s in pool(t)], n)
    if P.decide(a, c, .5): continue
    right = next((r for r, x in enumerate(g) if S(i, x[0]) == 1), None)
    why = "no answer" if not g else "share<.5" if c < .5 else "0/empty or unanswerable top group"
    rank[(why, "none" if right is None else "top1" if right == 0 else "top2" if right == 1 else "lower")] += 1
print("abstained, why x where the right answer is:"); [print("  ", k, v) for k, v in sorted(rank.items())]
# threshold sweep on the same pools
for th in (.25, .34, .4, .5, .6):
    c = Counter()
    for i in ids:
        t = tr[i]
        if t["final"].get("reason"): c[S(i, t["final"])] += 1; continue
        ps = [s["pred"] for s in pool(t)]; a, cf = P.vote(ps, len(ps)); c[S(i, a if P.decide(a, cf, th) else {"type": "abstain"})] += 1
    print(f"  threshold {th}: PA {(c[1]-c[-1])/len(ids):.3f} {c[1]}/{c[-1]}/{c[0]}")
# 2. plain vs thinking on questions that got thinking
c = Counter()
for i in ids:
    ss = tr[i]["samples"]
    if len(ss) <= 8: continue
    for name, part in (("plain", ss[:8]), ("think", ss[8:])):
        a, cf = P.vote([s["pred"] for s in part], len(part)); c[name, S(i, a) if a.get("type") != "abstain" else 0] += 1
print("hard questions, plurality of each pool (right/wrong):", {k: c[k, 1] for k in ("plain", "think")}, {k: c[k, -1] for k in ("plain", "think")})
# 3. context and ingest by split
def layers(run, b, kind):
    f = R / run / "logs" / f"_{kind}_{b}.json"
    return json.load(open(f, encoding="utf-8")) if f.exists() else {}
SYS, RB_T, RB_P = nt(P.SYSTEM), nt(open(R / "data/train/rulebook.md", encoding="utf-8").read()), nt(open(R / "data/public_test/rulebook.md", encoding="utf-8").read())
print(f"tokens: system prompt {SYS}, rulebook train {RB_T}, public {RB_P}")
def shop_stats(root, shops, run):
    out = []
    for b in shops:
        db, meta = load_business(Path(root), b)
        tabs = [t["name"] if isinstance(t, dict) and "name" in t else None for t in meta["tables"]]
        names = [r[0] for r in db.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='main'").fetchall()]
        cols = bad = empt = rows = 0
        for n in names:
            info = db.execute(f'DESCRIBE "{n}"').fetchall(); cnt = db.execute(f'SELECT COUNT(*) FROM "{n}"').fetchone()[0]; rows += cnt
            for c_ in info:
                col, typ = c_[0], c_[1]; cols += 1
                if col.startswith("_") or cnt == 0: continue
                nn, numlike = db.execute(f"""SELECT COUNT("{col}"), COUNT(TRY_CAST(regexp_replace(CAST("{col}" AS VARCHAR), '[ ,.%đ₫]', '', 'g') AS DOUBLE)) FROM "{n}" """).fetchone()
                empt += nn < cnt * .5
                if typ == "VARCHAR" and nn and numlike >= .9 * nn: bad += 1
        v = layers(run, b, "views"); d = layers(run, b, "debt")
        (llmviews.build(db, v["sql"]) if v.get("used") else None); vtxt = llmviews.describe(db, v["sql"], v["agree"], 8) if v.get("used") else ""; dtxt = debtview.describe(2, 8) if d.get("used") else ""
        out.append(dict(b=b, tables=len(names), cols=cols, rows=rows, num_text=bad, half_empty=empt, schema=nt(schema_text(meta)), readme=nt(meta["readme"]), view=nt(vtxt), debt=nt(dtxt)))
        db.close()
    return out
hs = sorted({trq[i]["business_id"] for i in ids}); ps = sorted({q["business_id"] for q in pub})
H = shop_stats("data/train", hs, "runs/sql_v2_holdout632"); Q = shop_stats("data/public_test", ps, "runs/sql_v2_public")
for name, S_ in (("holdout", H), ("public", Q)):
    print(name, {k: (round(st.median(x[k] for x in S_)), max(x[k] for x in S_)) for k in ("tables", "cols", "rows", "num_text", "half_empty", "schema", "readme", "view", "debt")}, "(median, max)")
agree = defaultdict(list)
for i, t in trp.items(): agree[t["business_id"]].append(t["conf"] if isinstance(t.get("conf"), (int, float)) else 0)
print("public shop: tables, schema tokens, prompt tokens, mean agreement")
for x in Q:
    print(f"  {x['b']}: {x['tables']} tables, schema {x['schema']}, prompt {SYS+RB_P+x['schema']+x['readme']+x['view']+x['debt']}, agreement {st.mean(agree[x['b']]):.2f}")
# 4. questions touching public-only rule terms
a_ = {l.strip() for l in open(R / "data/train/rulebook.md", encoding="utf-8")}; b_ = [l.strip() for l in open(R / "data/public_test/rulebook.md", encoding="utf-8") if l.strip()]
words = lambda s: [w for w in norm(s).split("_") if len(w) >= 3]
trainw = {w for l in a_ for w in words(l)}; trainbi = {x for l in a_ for x in zip(words(l), words(l)[1:])}
newbi = {x for l in b_ if l not in a_ for x in zip(words(l), words(l)[1:])} - trainbi
touch = [q["id"] for q in pub if set(zip(words(q["question"]), words(q["question"])[1:])) & newbi]
ct = lambda xs: (len(xs), round(st.mean(trp[i]["conf"] for i in xs), 3), round(sum(trp[i]["final"].get("type") == "abstain" for i in xs) / max(1, len(xs)), 3))
rest = [q["id"] for q in pub if q["id"] not in set(touch)]
print("public questions sharing a two-word term with public-only rule lines (n, mean agreement, abstain share):", ct(touch), "rest:", ct(rest), "| new two-word terms", len(newbi))
print("per shop share touching:", dict(sorted(Counter(trp[i]["business_id"] for i in touch).items())))
