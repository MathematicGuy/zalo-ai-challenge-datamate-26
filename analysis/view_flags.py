"""Label-free audit of the canonical views of every shop: python analysis/view_audit.py DATA_ROOT VIEWS_DIR [VIEWS_DIR ...] [--shops a,b]
For each _views_<shop>.json (the first dir that has it wins) builds the view and flags what the error analysis of 2026-11-05 found:
- ts: order_ts carries no time of day while a timestamp column of a table the view reads does (shift questions answer 0)
- split: an item whose lines are priced per qty at >= 2.5x or <= 1/2.5 of its median (a unit factor applied to base-unit lines, or not applied)
- cust: the view has no customer while a table it reads names customers; nocust: neither has any (customer questions are unanswerable)
- status: the raw tables it reads write cancel/return words but no line is cancelled or returned
- amount / qty: share of item lines with no or zero amount / qty, negative values
- dup: exact copies of a line (a join fans out); no view: no draft passed"""
import argparse, json, re, sys
from pathlib import Path
R = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(R), str(R / "analysis")]
from datamate import ingest, views as llmviews

ap = argparse.ArgumentParser()
ap.add_argument("root", type=Path); ap.add_argument("views", nargs="+", type=Path); ap.add_argument("--shops")
a = ap.parse_args()
shops = sorted(p.name for p in (a.root / "data").iterdir() if p.is_dir())
if a.shops:
    shops = [s for s in shops if s in a.shops.split(",")]
q = lambda n: '"' + n.replace('"', '""') + '"'
CUST = re.compile(r"(?i)khach|khách|customer|thành viên|thanh vien|member|người mua|nguoi mua")
WALKIN = "('khách lẻ','khach le','kl','lẻ','le','vãng lai','vang lai','khách vãng lai','khach vang lai','')"
STATUS = r"(?i)(h[uủ]y|hu[ỷy]|hoàn|hoan|trả|tra hang|đổi trả|doi tra|không giao|khong giao|returned|cancel)"
print("shop      lines  view  agree  flags")
for b in shops:
    f = next((d / "logs" / f"_views_{b}.json" for d in a.views if (d / "logs" / f"_views_{b}.json").exists()), None)
    if f is None:
        print(f"{b:<9} {'':>6}  -     -      no file"); continue
    d = json.loads(f.read_text(encoding="utf-8"))
    flags = []
    if not d.get("sql") or not d.get("used", True):
        flags.append(f"NO VIEW ({d.get('agree')}/{d.get('k')} agree; {len(d.get('errors') or [])} errors)")
    r = ingest.load_business(a.root, b, canon=True); db = r[0] if isinstance(r, tuple) else r
    if not d.get("sql"):
        print(f"{b:<9} {'':>6}  no    {d.get('agree')}/{d.get('k')}  " + "; ".join(flags)); continue
    sql = d["sql"]
    tabs = [t for (t,) in db.execute("SELECT table_name FROM information_schema.tables").fetchall()]
    used = [t for t in tabs if re.search(r'(?i)"' + re.escape(t) + r'"|\b' + re.escape(t) + r'\b', sql) and t != "calendar"]
    try:
        llmviews.build(db, sql)
    except Exception as e:
        print(f"{b:<9} {'':>6}  err   {d.get('agree')}/{d.get('k')}  build: {str(e)[:120]}"); continue
    V = "v_sales_all"
    n, ts_n, ts_time = db.execute(f"SELECT count(*), count(order_ts), count(*) FILTER (WHERE order_ts IS NOT NULL AND CAST(order_ts AS TIME) <> TIME '00:00:00') FROM {V}").fetchone()
    # ts
    if ts_time < 0.05 * n:
        cand = []
        for t in used:
            for c, ty in db.execute(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{t}'").fetchall():
                if "TIMESTAMP" in ty.upper() or ty.upper() in ("TIME", "VARCHAR"):
                    try:
                        if "TIMESTAMP" in ty.upper():
                            k, m = db.execute(f"SELECT count(*) FILTER (WHERE CAST({q(c)} AS TIME) <> TIME '00:00:00'), count({q(c)}) FROM {q(t)}").fetchone()
                        elif ty.upper() == "TIME":
                            k, m = db.execute(f"SELECT count({q(c)}), count(*) FROM {q(t)}").fetchone()
                        else:
                            k, m = db.execute(f"SELECT count(*) FILTER (WHERE regexp_matches({q(c)}, '^\\s*\\d{{1,2}}:\\d{{2}}')), count({q(c)}) FROM {q(t)}").fetchone()
                        if m and k > 0.5 * m:
                            cand.append(f"{t}.{c}")
                    except Exception:
                        pass
        if cand:
            flags.append(f"TS: order_ts has a time on {ts_time}/{n} lines; times in {', '.join(cand[:3])}")
    # split
    rows = db.execute(f"""WITH s AS (SELECT product_name p, amount / qty pr FROM {V} WHERE qty > 0 AND amount > 0 AND product_name IS NOT NULL AND line_kind = 'item' AND product_name NOT ILIKE '%phí%'),
        m AS (SELECT p, median(pr) md, count(*) n FROM s GROUP BY p HAVING count(*) >= 10),
        o AS (SELECT s.p, m.n, count(*) FILTER (WHERE s.pr >= 2.5 * m.md OR s.pr <= m.md / 2.5) k, round(max(s.pr) / min(s.pr), 1) r FROM s JOIN m USING (p) GROUP BY s.p, m.n)
        SELECT p, n, k, r FROM o WHERE k >= 3 ORDER BY k DESC""").fetchall()
    if rows:
        flags.append(f"SPLIT: {len(rows)} items, {sum(x[2] for x in rows)} lines: " + "; ".join(f"{x[0][:40]} x{x[3]} ({x[2]}/{x[1]})" for x in rows[:3]))
    # customers
    vc = db.execute(f"SELECT count(*) FILTER (WHERE customer_key IS NOT NULL) FROM {V}").fetchone()[0]
    raw = []
    for t in used:
        for c, ty in db.execute(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{t}'").fetchall():
            if CUST.search(c):
                try:
                    k = db.execute(f"SELECT count(*) FILTER (WHERE {q(c)} IS NOT NULL AND lower(trim(CAST({q(c)} AS VARCHAR))) NOT IN {WALKIN}) FROM {q(t)}").fetchone()[0]
                except Exception:
                    k = 0
                if k:
                    raw.append((f"{t}.{c}", k))
    if vc == 0 and raw:
        flags.append(f"CUST: no customer in the view; raw names them in {', '.join(x for x, _ in raw[:3])}")
    elif vc == 0:
        flags.append("nocust: no customer recorded")
    elif vc < 0.02 * n:
        flags.append(f"cust low: {vc}/{n} lines have a customer" + (f"; raw {raw[0][0]} {raw[0][1]}" if raw else ""))
    # status
    canc, ret = db.execute(f"SELECT count(*) FILTER (WHERE is_cancelled), count(*) FILTER (WHERE is_returned) FROM {V}").fetchone()
    if canc + ret == 0:
        hits = []
        for t in used:
            for (c,) in db.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{t}' AND data_type = 'VARCHAR'").fetchall():
                try:
                    k = db.execute(f"SELECT count(*) FROM {q(t)} WHERE regexp_matches({q(c)}, '{STATUS}') AND length({q(c)}) < 40").fetchone()[0]
                except Exception:
                    k = 0
                if k:
                    hits.append(f"{t}.{c} ({k})")
        if hits:
            flags.append(f"STATUS: no cancelled/returned line; raw has status words in {', '.join(hits[:3])}")
    # amount / qty / dup
    am0, q0, neg, items = db.execute(f"SELECT count(*) FILTER (WHERE amount IS NULL OR amount = 0), count(*) FILTER (WHERE qty IS NULL OR qty = 0), count(*) FILTER (WHERE amount < 0 OR qty < 0), count(*) FROM {V} WHERE line_kind = 'item'").fetchone()
    if items and am0 > 0.05 * items:
        flags.append(f"amount: {am0}/{items} item lines with no or zero amount")
    if items and q0 > 0.02 * items:
        flags.append(f"qty: {q0}/{items} item lines with no or zero qty")
    if items and neg > 0.01 * items:
        flags.append(f"neg: {neg}/{items} item lines negative")
    sp = llmviews.unit_spread(db, sql)
    if sp[0] and sp[0] > 1.5:
        flags.append(f"unit spread {sp[0]:.1f} over {sp[1]} items")
    print(f"{b:<9} {n:>6}  {'yes' if d.get('used', True) else 'no':<4}  {d.get('agree')}/{d.get('k')}  " + ("; ".join(flags) if flags else "ok"), flush=True)
