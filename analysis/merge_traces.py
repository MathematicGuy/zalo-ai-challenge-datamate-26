"""Trace of run A with the questions run B holds replaced by B's. python analysis/merge_traces.py A B OUT"""
import json
import sys
from pathlib import Path

a, b, out = map(Path, sys.argv[1:4])
new = {json.loads(x)["id"]: x for x in open(b / "trace.jsonl", encoding="utf-8")}
out.mkdir(parents=True, exist_ok=True)
n = 0
with open(out / "trace.jsonl", "w", encoding="utf-8") as f:
    for x in open(a / "trace.jsonl", encoding="utf-8"):
        i = json.loads(x)["id"]
        n += i in new
        f.write(new.get(i, x))
print(f"{n} of {len(new)} questions replaced -> {out}")
