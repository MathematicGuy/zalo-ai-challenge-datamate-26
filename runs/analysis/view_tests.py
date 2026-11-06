"""Run the view data tests (catalog, time, split) on saved views: python runs/analysis/view_tests.py DATA_ROOT VIEWS_DIR [VIEWS_DIR ...]"""
import json, sys
from pathlib import Path
R = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(R / "sqlbase")]
import ingest, llmviews
root, dirs = Path(sys.argv[1]), [Path(x) for x in sys.argv[2:]]
for b in sorted(p.name for p in (root / "data").iterdir() if p.is_dir()):
    f = next((d / "logs" / f"_views_{b}.json" for d in dirs if (d / "logs" / f"_views_{b}.json").exists()), None)
    if not f or not json.loads(f.read_text())["sql"]:
        print(b, "no view"); continue
    sql = json.loads(f.read_text())["sql"]
    r = ingest.load_business(root, b, canon=True); db = r[0] if isinstance(r, tuple) else r
    llmviews.build(db, sql)
    out = []
    for name, fn in (("catalog", lambda: llmviews.catalog_test(db, "v_")), ("time", lambda: llmviews.time_test(db, "v_", sql)), ("split", lambda: llmviews.split_test(db, "v_"))):
        try:
            fn()
        except ValueError as e:
            out.append(f"{name}: {str(e)[:150]}")
    print(b, "; ".join(out) if out else "pass", flush=True)
