"""Writes a submission from a run: debt routing and the vote, then the reviewer's picks on questions the vote leaves unanswered. python runs/analysis/apply_review.py RUN REVIEW.jsonl OUT --data data/public_test"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

R = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(R / "sqlbase"), str(R / "train")]
import predict as P
from check_submission import problem

p = argparse.ArgumentParser()
p.add_argument("run")
p.add_argument("review")
p.add_argument("out")
p.add_argument("--data", default="data/public_test")
p.add_argument("--answer-unanswerable", action="store_true", help="answer unanswerable when that is the voted group at share >= .5")
a = p.parse_args()
rv = {r["id"]: r for r in map(json.loads, open(R / a.review, encoding="utf-8"))} if a.review != "none" else {}
qs = [json.loads(x) for x in open(R / a.data / "questions.jsonl", encoding="utf-8-sig") if x.strip()]
res, c = {}, Counter()
for line in open(R / a.run / "trace.jsonl", encoding="utf-8"):
    t = json.loads(line)
    i = t["id"]
    if t["final"].get("reason"):
        res[i] = t["final"]
        continue
    idx = P.debt_pool([s["text"] for s in t["samples"]], 8)
    ans, conf = P.vote([t["samples"][j]["pred"] for j in idx], len(idx))
    r = ans if P.decide(ans, conf, 0.5) else {"type": "abstain"}
    if a.answer_unanswerable and ans.get("type") == "unanswerable" and conf >= 0.5:
        r = ans
        c["unanswerable"] += 1
    if r["type"] == "abstain" and rv.get(i, {}).get("choice"):
        ch = rv[i]["choice"]
        pick = next((x["pred"] for x in rv[i]["candidates"] if x["label"] == ch), None)
        if pick and P.decide(pick, 1.0, 0.5):
            r = pick
            c["rescued"] += 1
        else:
            c["reviewer gave no answer"] += 1
    res[i] = {"type": "abstain"} if problem({"id": i, **r}) else r
out = R / a.out
out.mkdir(parents=True, exist_ok=True)
P.write_submission(out / "submission.jsonl", qs, res)
print(dict(c), "| answered", sum(x["type"] != "abstain" for x in res.values()), "of", len(qs), "->", out)
