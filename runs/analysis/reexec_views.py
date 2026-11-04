"""Re-runs the saved SQL of a run against other views, without new samples: samples reading the views get new answers, the others keep theirs.
python runs/analysis/reexec_views.py RUN NEW_VIEWS_DIR OUT --data data/train --debts-from DIR"""
import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

R = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(R / "sqlbase")]
import debtview
import llmviews
import predict as P
from ingest import load_business

VIEWS = re.compile(r'(?i)\bv_(sales|orders)(_all)?\b')


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("views")
    p.add_argument("out")
    p.add_argument("--data", default="data/train")
    p.add_argument("--debts-from")
    p.add_argument("--canon", action="store_true", help="load the shops with one spelling per name, and re-run every sample, raw tables too")
    a = p.parse_args()
    data, views, out = R / a.data, R / a.views / "logs", R / a.out
    debts = R / (a.debts_from or a.views) / "logs"
    out.mkdir(parents=True, exist_ok=True)
    tr = defaultdict(list)
    for line in open(R / a.run / "trace.jsonl", encoding="utf-8"):
        t = json.loads(line)
        tr[t["business_id"]].append(t)
    t0, c = time.time(), Counter()
    with open(out / "trace.jsonl", "w", encoding="utf-8") as f:
        for b, ts in sorted(tr.items()):
            db, meta = load_business(data, b, canon=a.canon)
            db.execute("SET memory_limit = '6GB'")
            v = json.loads((views / f"_views_{b}.json").read_text(encoding="utf-8"))
            if v.get("sql"):
                llmviews.build(db, v["sql"])
            df = debts / f"_debt_{b}.json"
            d = json.loads(df.read_text(encoding="utf-8")) if df.exists() else {}
            if d.get("sqls") and d.get("used"):
                debtview.build(db, d["sqls"])
            for t in ts:
                for s in t["samples"]:
                    last = s["text"].split("\n#REPAIR\n")[-1]
                    if not s["pred"] or not (a.canon or VIEWS.search(last)):
                        continue
                    c["reran"] += 1
                    try:
                        new = P.execute(db, last, t["question"])
                    except Exception as e:
                        c["now fails"] += 1
                        s["pred"], s["error"] = None, str(e)[:300]
                        continue
                    if new != s["pred"]:
                        c["changed"] += 1
                    s["pred"] = new
                f.write(json.dumps(t, ensure_ascii=False, default=str) + "\n")
            f.flush()
            db.close()
            print(f"[{b}] {len(ts)} q, {time.time() - t0:.0f}s, {c.most_common()}", flush=True)


if __name__ == "__main__":
    main()
