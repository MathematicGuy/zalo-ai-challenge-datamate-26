"""Invariants of saved canonical views, checked against the raw tables (no labels, no LLM).
Usage: python runs/analysis/view_audit.py DATA_SPLIT RUN_OR_CONFIG   (reads RUN/logs/_views_<shop>.json or RUN/shard*/logs/...)
Per shop: rows, orders, the share of lines that are copies on the line's own fields (a lookup join fan-out the exact-copy
test misses), orders with 2+ customers, item lines with no/neg qty or amount, and the cancelled/returned order counts of
the view against the status flags ingest derives from the raw status column."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sqlbase"))
import llmviews  # noqa: E402
from ingest import load_business, quote  # noqa: E402


def raw_status(db, meta):
    """(table, cancelled orders, returned orders) of the biggest table ingest flagged, or None."""
    best = None
    for t in meta["tables"]:
        cols = {c for c, *_ in t["columns"]}
        if {"_don_huy", "_don_tra"} <= cols and (best is None or t["rows"] > best["rows"]):
            best = t
    if not best:
        return None
    from ingest import ORDER_ID, ckey  # noqa: E402
    import re
    oid = next(c for c, *_ in best["columns"] if re.search(ORDER_ID, ckey(c)))
    c, r = db.execute(f'SELECT count(DISTINCT {quote(oid)}) FILTER (WHERE "_don_huy"), count(DISTINCT {quote(oid)}) FILTER (WHERE "_don_tra" AND NOT "_don_huy") FROM {quote(best["name"])}').fetchone()
    return best["name"], c, r


def audit(db, meta):
    q = lambda s: db.execute(s).fetchone()
    rows, orders = q("SELECT count(*), count(DISTINCT order_id) FROM v_sales_all")
    near = rows - q("SELECT count(*) FROM (SELECT DISTINCT order_id, order_ts, product_name, qty, amount, line_kind, line_cancelled, line_returned FROM v_sales_all)")[0]
    multi = q("SELECT count(*) FROM (SELECT order_id FROM v_sales_all GROUP BY 1 HAVING count(DISTINCT customer_key) > 1)")[0]
    badq, bada, zero = q("SELECT count(*) FILTER (WHERE qty IS NULL OR qty <= 0), count(*) FILTER (WHERE amount < 0), count(*) FILTER (WHERE amount = 0 OR amount IS NULL) FROM v_sales_all WHERE line_kind = 'item'")
    can, ret = q("SELECT count(*) FILTER (WHERE is_cancelled), count(*) FILTER (WHERE is_returned AND NOT is_cancelled) FROM v_orders_all")
    return dict(rows=rows, orders=orders, near_dup=near, multi_cust=multi, item_bad_qty=badq, item_neg_amt=bada, item_zero_amt=zero,
                view_cancel=can, view_return=ret, raw=raw_status(db, meta))


def main():
    data, run = Path(sys.argv[1]), Path(sys.argv[2])
    files = sorted(run.glob("logs/_views_*.json")) + sorted(run.glob("shard*/logs/_views_*.json"))
    for f in files:
        b = f.stem.removeprefix("_views_")
        v = json.loads(f.read_text(encoding="utf-8"))
        if not v.get("sql") or not v.get("used"):
            print(b, "no view used")
            continue
        db, meta = load_business(data, b, canon=True)
        llmviews.build(db, v["sql"])
        print(b, json.dumps(audit(db, meta), ensure_ascii=False))
        db.close()


if __name__ == "__main__":
    main()
