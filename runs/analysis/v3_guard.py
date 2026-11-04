"""Catches a broken generation run before hours of training: same questions, labels from train answers, against older runs of the same LoRA.
python runs/analysis/v3_guard.py RUN REF [REF ...]; exit 1 when RUN is more than 0.05 below the references. Not a measure of progress: LoRA v2 has seen these shops."""
import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(R / "src")]
from datamate.scoring import score_one

gold = {x["id"]: x for x in map(json.loads, open(R / "data/train/answers.jsonl"))}
def finals(run):
    return {t["id"]: t["final"] for t in map(json.loads, open(R / run / "trace.jsonl", encoding="utf-8"))}
new, ref = finals(sys.argv[1]), {}
for r in sys.argv[2:]:
    ref.update(finals(r))
ids = sorted(set(new) & set(ref))
pa = lambda f: sum(0 if f[i].get("type") == "abstain" else score_one({"id": i, **f[i]}, gold[i]) for i in ids) / max(len(ids), 1)
a, b = pa(new), pa(ref)
print(f"{len(ids)} shared questions: this run {a:.3f}, reference {b:.3f}; all {len(new)} questions answered {sum(f.get('type') != 'abstain' for f in new.values())}")
sys.exit(0 if ids and a >= b - 0.05 else 1)
