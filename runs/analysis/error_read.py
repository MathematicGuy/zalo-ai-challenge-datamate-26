"""Prints the wrong final answers of the current setup on the holdout for reading, or the abstained ones with --abstain. python runs/analysis/error_read.py [--abstain] [--max N]"""
import json, re, sys
from collections import Counter
from pathlib import Path
R = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(R / "sqlbase"), str(R / "src")]
import predict as P, fewshot
from datamate.scoring import score_one
gold = {x["id"]: x for x in map(json.loads, open(R / "data/train/answers.jsonl"))}
tr = {json.loads(l)["id"]: json.loads(l) for l in open(R / "runs/sql_v2_holdout632/trace.jsonl")}
mode = -1 if "--abstain" not in sys.argv else 0
limit = int(sys.argv[sys.argv.index("--max") + 1]) if "--max" in sys.argv else 60
asof = {q["id"]: q["as_of_date"] for q in map(json.loads, open(R / "data/train/questions.jsonl", encoding="utf-8-sig"))}
n = 0
for i, t in sorted(tr.items(), key=lambda kv: kv[1]["business_id"]):
    if t["final"].get("reason"): continue
    ss = t["samples"]; idx = P.debt_pool([s["text"] for s in ss], 8); voters = [ss[j] for j in idx]
    a, c = P.vote([s["pred"] for s in voters], len(voters)); r = a if P.decide(a, c, .5) else {"type": "abstain"}
    if score_one({"id": i, **r}, gold[i]) != mode or n >= limit: continue
    n += 1
    g = gold[i]
    pool = [s for s in voters if s["pred"] and P.answer_key(s["pred"]) == P.answer_key(a)] if a.get("type") != "abstain" else []
    sql = min((P.parse(s["text"].split("\n#REPAIR\n")[-1])[1] or "" for s in pool), key=len, default="")
    right = next((s for s in ss if s["pred"] and score_one({"id": i, **s["pred"]}, g) == 1), None)
    print(f"--- {t['business_id']} {i} asof {asof[i]} share {c:.2f}")
    print("Q:", t["question"])
    print("GOLD:", g["type"], json.dumps(g.get("answer"), ensure_ascii=False)[:230])
    print("VOTE:", a.get("type"), json.dumps(a.get("answer"), ensure_ascii=False, default=str)[:230])
    print("SQL:", re.sub(r"\s+", " ", sql)[:520])
    if right: print("RIGHT SAMPLE SQL:", re.sub(r"\s+", " ", P.parse(right["text"].split("\n#REPAIR\n")[-1])[1] or "")[:420])
print("printed", n)
