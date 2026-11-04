"""Which rulebook sections go with weak answers. A question is linked to the sections sharing its rarer terms (coarse). Aggregates here; headings only in the local rule_sections.txt."""
import json, math, re, sys, statistics as st
from collections import Counter, defaultdict
from pathlib import Path
R = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(R / "sqlbase"), str(R / "src")]
import predict as P
from ingest import norm
from datamate.scoring import score_one
from tokenizers import Tokenizer
tok = Tokenizer.from_file(str(R / "sqlbase/models/Qwen3.5-9B/tokenizer.json")); nt = lambda s: len(tok.encode(s or "").ids)
rd = lambda p: [json.loads(l) for l in open(p, encoding="utf-8-sig") if l.strip()]
gold = {x["id"]: x for x in map(json.loads, open(R / "data/train/answers.jsonl"))}
trq = {q["id"]: q for q in rd(R / "data/train/questions.jsonl")}
ids = [i.strip() for i in open(R / "splits/holdout600.txt") if i.strip()]
H = {json.loads(l)["id"]: json.loads(l) for l in open(R / "runs/sql_v2_holdout632/trace.jsonl")}
Q = {json.loads(l)["id"]: json.loads(l) for l in open(R / "runs/sql_v2_public/trace.jsonl")}
pub = rd(R / "data/public_test/questions.jsonl")
def sections(path):
    secs = []
    for l in open(path, encoding="utf-8"):
        l = l.rstrip("\n")
        if re.match(r"#+\s", l): secs.append({"head": l.strip(), "lines": []})
        elif secs and l.strip(): secs[-1]["lines"].append(l.strip())
    return secs
words = lambda s: [w for w in norm(s).split("_") if len(w) >= 2]
terms = lambda s: (lambda w: set(w) | {a + " " + b for a, b in zip(w, w[1:])})(words(s))
def linker(secs):
    F = [terms(s["head"] + " " + " ".join(s["lines"])) for s in secs]; df = Counter(t for f in F for t in f); n = len(F)
    idf = {t: math.log(n / v) for t, v in df.items()}   # a term in every section counts for nothing
    def link(q, k=2, need=2.0):
        qt = terms(q); sc = [(sum(idf[t] for t in qt & f), j) for j, f in enumerate(F)]
        return [j for s, j in sorted(sc, reverse=True)[:k] if s >= need]
    return link
MAC = re.compile(r"debt_(at|new|repaid)\s*\(")
use = lambda s: bool(MAC.search(s["text"].split("\n#REPAIR\n")[-1]))
def final(t):
    if t["final"].get("reason"): return t["final"], 1.0
    ss = t["samples"]; th = ss[8:]
    if th and sum(map(use, th)) / len(th) >= .5: ss = [s for s in ss if use(s)]
    ps = [s["pred"] for s in ss]; a, c = P.vote(ps, len(ps)); return (a if P.decide(a, c, .5) else {"type": "abstain"}), c
trainT = {l.strip() for l in open(R / "data/train/rulebook.md", encoding="utf-8")}
PS, TS = sections(R / "data/public_test/rulebook.md"), sections(R / "data/train/rulebook.md")
lp, lt = linker(PS), linker(TS)
head_map = {norm(s["head"]): i for i, s in enumerate(PS)}
pub_stat, hold_stat = defaultdict(list), defaultdict(list)
for q in pub:
    a, c = final(Q[q["id"]])
    for j in lp(q["question"]): pub_stat[j].append((c, a.get("type") == "abstain"))
nolink = sum(not lp(q["question"]) for q in pub)
for i in ids:
    a, c = final(H[i]); s = score_one({"id": i, **a}, gold[i])
    for j in lt(trq[i]["question"]):
        k = head_map.get(norm(TS[j]["head"]))
        hold_stat[k if k is not None else f"t{j}"].append(s)
with open(R / "runs/analysis/rule_sections.txt", "w", encoding="utf-8") as f:
    for i, s in enumerate(PS): f.write(f"{i}\t{s['head']}\n")
print(f"public questions with no linked section: {nolink}")
print("sec tokens newlines | public n agreement abstain | holdout n PA")
for i, s in enumerate(PS):
    v = pub_stat[i]; h = hold_stat[i]
    print(f"{i:3d} {nt(chr(10).join(s['lines'])):5d} {sum(l not in trainT for l in s['lines']):3d} | {len(v):4d} {st.mean(x[0] for x in v) if v else 0:.2f} {sum(x[1] for x in v)/max(1,len(v)):4.0%} | {len(h):4d} {(h.count(1)-h.count(-1))/max(1,len(h)):+.2f}")
print("train-only sections on holdout:", {k: (len(v), round((v.count(1)-v.count(-1))/len(v), 2)) for k, v in hold_stat.items() if isinstance(k, str)})
