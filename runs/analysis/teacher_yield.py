"""Percent of a run's train questions with a right sample: python runs/analysis/teacher_yield.py RUN"""
import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(R / "src")]
from datamate.scoring import score_one

gold = {x["id"]: x for x in map(json.loads, open(R / "data/train/answers.jsonl"))}
n = hit = 0
for line in open(R / sys.argv[1] / "trace.jsonl", encoding="utf-8"):
    t = json.loads(line)
    n += 1
    hit += any(s.get("pred") and score_one({"id": t["id"], **s["pred"]}, gold[t["id"]]) == 1 for s in t["samples"])
print(round(100 * hit / max(n, 1)))
