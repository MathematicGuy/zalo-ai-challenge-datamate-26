"""Questions no 9B sample solved, in chunks for the teacher model, and its views: python runs/analysis/v3_hard_ids.py RUN [RUN ...]. Train data only, labels from train answers."""
import json
import random
import shutil
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(R / "src")]
from datamate.scoring import score_one

gold = {x["id"]: x for x in map(json.loads, open(R / "data/train/answers.jsonl"))}
hard, total = [], 0
for run in sys.argv[1:]:
    for line in open(R / run / "trace.jsonl", encoding="utf-8"):
        t = json.loads(line)
        total += 1
        if not any(s.get("pred") and score_one({"id": t["id"], **s["pred"]}, gold[t["id"]]) == 1 for s in t["samples"]):
            hard.append(t["id"])
random.Random(0).shuffle(hard)  # chunks mix the shops, so a time limit does not favour the first shops
for f in (R / "runs/analysis").glob("v3_teacher_ids_*.txt"):
    f.unlink()
for i in range(0, len(hard), 40):
    (R / f"runs/analysis/v3_teacher_ids_{i // 40:02d}.txt").write_text("\n".join(hard[i:i + 40]) + "\n")
dst = R / "runs/views_v3_teacher/logs"
dst.mkdir(parents=True, exist_ok=True)
for src in [R / "runs/views_qty_holdout/logs", R / "runs/sql_v5_gen24/logs"]:
    for f in list(src.glob("_views_*.json")) + list(src.glob("_debt_*.json")):
        if f.name.startswith("_views_"):
            v = json.loads(f.read_text(encoding="utf-8"))
            if not v.get("used"):
                v["sql"] = None  # at k=2 the agreement test would let a rejected view back in
            v.setdefault("drafts", [])
            (dst / f.name).write_text(json.dumps(v, ensure_ascii=False, indent=1), encoding="utf-8")
        else:
            shutil.copy(f, dst / f.name)
print(f"{len(hard)} of {total} questions have no right 9B sample, {len(list((R / 'runs/analysis').glob('v3_teacher_ids_*.txt')))} chunks")
