"""Error analysis on train labels: type confusion, question intent, period and noise, how answers are wrong, seen against unseen shops, what the LoRA rows teach, what each adapter changed.
Public questions appear only as shares and agreement. python runs/analysis/confusion.py"""
import json
import re
import statistics as st
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

R = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(R / "sqlbase"), str(R / "src")]
import fewshot
import predict as P
from datamate.scoring import score_one
from evaluation import score as score2
from ingest import norm

rd = lambda p: [json.loads(x) for x in open(R / p, encoding="utf-8-sig") if x.strip()]
gold = {x["id"]: x for x in rd("data/train/answers.jsonl")}
Q = {q["id"]: q for q in rd("data/train/questions.jsonl")}
ids = [i.strip() for i in open(R / "splits/holdout600.txt") if i.strip()]
hold_shops = {Q[i]["business_id"] for i in ids}
load = lambda r: {json.loads(x)["id"]: json.loads(x) for x in open(R / r / "trace.jsonl", encoding="utf-8")}
S = lambda i, r: score_one({"id": i, **(r or {"type": "abstain"})}, gold[i])
last = lambda s: s["text"].split("\n#REPAIR\n")[-1]


def final(t, start=8):
    if t["final"].get("reason"):
        return t["final"], 1.0
    ss = t["samples"]
    idx = P.debt_pool([s["text"] for s in ss], start)
    a, c = P.vote([ss[j]["pred"] for j in idx], len(idx))
    return (a if P.decide(a, c, .5) else {"type": "abstain"}), c


INTENTS = [("delivery", r"dot_hang|dot_giao|giao_hang|xe_cho|xe_da_giao|xe_giao"), ("debt", r"(^|_)no(_|$)|cong_no|mua_chiu|ghi_no"), ("lot/expiry", r"(^|_)lo(_|$)|het_han|han_dung"),
           ("stock", r"(^|_)ton(_|$)|ton_kho|het_hang|khong_ban_duoc"), ("price change", r"tang_gia|len_gia|gia_ban|doi_gia"), ("cost/profit", r"(^|_)lai(_|$)|loi_nhuan|gia_von|nhap_hang|tien_nhap|nha_cung_cap|chi_phi"),
           ("ranking", r"ban_chay|ban_nhieu|hot_nhat|(^|_)top(_|$)|nhieu_nhat|cao_nhat|lon_nhat|it_nhat|hang_nao|mat_hang_nao"), ("percent", r"phan_tram|ti_le|ty_le|ti_trong|ty_trong|tang_truong|tang_hay_giam|(^|_)%"),
           ("quantity", r"so_luong|bao_nhieu_(cai|ly|lon|vien|hop|chai|bao|goi|chiec|dia|phan|kg|thung)|(bn|bnhiu)_(cai|ly|lon|vien|hop|chai|bao)|so_ly|so_cai"),
           ("order count", r"so_don|bao_nhieu_don|(bn|bnhiu)_don|so_bill|so_hoa_don|may_don|bao_nhieu_bill"), ("customer", r"khach"), ("revenue", r"doanh_thu|dthu|(^|_)tien(_|$)|thu_ve|ban_duoc_bao_nhieu")]
PERIODS = [("today", r"hom_nay|hnay"), ("yesterday", r"hom_qua|bua_qua|hqua|bqua"), ("this week", r"tuan_nay"), ("last week", r"tuan_truoc|tuan_roi"), ("this month", r"thang_nay"), ("last month", r"thang_truoc|thang_roi"),
           ("N days", r"\d+_ngay"), ("year to date", r"tu_dau_nam|dau_nam"), ("tet", r"(^|_)tet(_|$)"), ("quarter", r"(^|_)quy_"), ("month/year", r"thang_\d+"), ("explicit day", r"\d+_\d+_\d{4}|ngay_\d+")]
intent = lambda q: next((n for n, p in INTENTS if re.search(p, norm(q))), "other")
period = lambda q: next((n for n, p in PERIODS if re.search(p, norm(q))), "none named")
ascii_only = lambda q: all(ord(c) < 128 for c in q)
pa = lambda v: (v.count(1) - v.count(-1)) / max(1, len(v))


def table(title, keyf, xs, base, after=None, extra=None):
    print(f"\n## {title}")
    groups = defaultdict(list)
    for i in xs:
        groups[keyf(i)].append(i)
    for k, g in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        c = Counter(base[i] for i in g)
        line = f"  {k:16s} n={len(g):4d} PA {pa([base[i] for i in g]):+.2f} {c[1]}/{c[-1]}/{c[0]}"
        if after:
            c2 = Counter(after[i] for i in g)
            line += f" | fixed {pa([after[i] for i in g]):+.2f} {c2[1]}/{c2[-1]}/{c2[0]}"
        if extra:
            line += " | " + extra(k)
        print(line)


H, F = load("runs/sql_v2_holdout632"), load("runs/sql_v2_qtyfix632")
final_h = {i: final(H[i]) for i in ids}
final_f = {i: final(F[i]) for i in ids}
base = {i: S(i, final_h[i][0]) for i in ids}
after = {i: S(i, final_f[i][0]) for i in ids}
print("holdout", Counter(base.values()), "fixed", Counter(after.values()))

# A. type confusion
print("\n## A. Type confusion: gold type (rows) x final type on the fixed views (counts; right in parentheses)")
kinds = ["number", "percent", "string", "date", "boolean", "list", "table", "unanswerable", "abstain"]
print("  " + " " * 13 + " ".join(f"{k[:7]:>9s}" for k in kinds))
m = Counter(); mr = Counter()
for i in ids:
    g, a = gold[i]["type"], final_f[i][0]["type"]
    m[g, a] += 1
    mr[g, a] += after[i] == 1
for g in kinds[:-1]:
    print(f"  {g:13s}" + " ".join(f"{(str(m[g, a]) + ('(' + str(mr[g, a]) + ')' if mr[g, a] else '')) if m[g, a] else '.':>9s}" for a in kinds))
sm = Counter()
for i in ids:
    for s in F[i]["samples"]:
        sm[gold[i]["type"], (s["pred"] or {}).get("type", "no answer")] += 1
print("  samples of the wrong type, by gold type:", {g: f"{sum(v for (x, y), v in sm.items() if x == g and y not in (g, 'no answer'))}/{sum(v for (x, y), v in sm.items() if x == g)}" for g in kinds[:-1]})
print("  samples with no answer, by gold type:", {g: f"{sm[g, 'no answer']}/{sum(v for (x, y), v in sm.items() if x == g)}" for g in kinds[:-1]})

# B. intent, with sample accuracy by source
pub = rd("data/public_test/questions.jsonl")
PT = load("runs/sql_v2_public")
pub_i = defaultdict(list)
for q in pub:
    pub_i[intent(q["question"])].append(final(PT[q["id"]]))
tables_of = lambda s: fewshot.tables_of(last(s).split("</think>")[-1])
kind = lambda s: (lambda x: "views" if x and x <= fewshot.CANON else "raw" if x and not (x & fewshot.CANON) else "mixed" if x else "none")(tables_of(s))
src = defaultdict(lambda: defaultdict(list))
for i in ids:
    for s in F[i]["samples"][:8]:
        if s["pred"]:
            src[intent(Q[i]["question"])][kind(s)].append(S(i, s["pred"]) == 1)


def extra_intent(k):
    p = pub_i.get(k, [])
    v, r = src[k]["views"], src[k]["raw"]
    return (f"public {len(p):3d} q, abstain {sum(a['type'] == 'abstain' for a, _ in p) / max(1, len(p)):4.0%} agree {st.mean(c for _, c in p) if p else 0:.2f} | "
            f"plain samples right: views {sum(v) / max(1, len(v)):4.0%} of {len(v)}, raw {sum(r) / max(1, len(r)):4.0%} of {len(r)}")


table("B. Question intent (keywords)", lambda i: intent(Q[i]["question"]), ids, base, after, extra_intent)

# C. period and noise
pub_p = Counter(period(q["question"]) for q in pub)
table("C1. Period named", lambda i: period(Q[i]["question"]), ids, base, after, lambda k: f"public {pub_p[k]} q")
pub_a = Counter("no accents" if ascii_only(q["question"]) else "accents" for q in pub)
table("C2. Writing", lambda i: "no accents" if ascii_only(Q[i]["question"]) else "accents", ids, base, after, lambda k: f"public {pub_a[k]} q")
slang = lambda q: bool(set(re.findall(r"\w+", q.lower())) & set(P.SLANG))
pub_s = Counter("abbreviated" if slang(q["question"]) else "plain" for q in pub)
table("C3. Abbreviations", lambda i: "abbreviated" if slang(Q[i]["question"]) else "plain", ids, base, after, lambda k: f"public {pub_s[k]} q")

# D. how answers are wrong, on the fixed views
print("\n## D. How final answers are wrong, on the fixed views")
modes = Counter()
plain = lambda a: unicodedata.normalize("NFKD", str(a)).encode("ascii", "ignore").decode().lower().strip()
for i in ids:
    if after[i] != -1:
        continue
    r, g = final_f[i][0], gold[i]
    a, b = r.get("answer"), g.get("answer")
    if r["type"] != g["type"]:
        modes[f"wrong type {g['type']} -> {r['type']}"] += 1
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if b and a and round(a / b) == a / b and abs(a / b) >= 2:
            modes["number: exact multiple of gold"] += 1
        elif b and a and round(b / a) == b / a and abs(b / a) >= 2:
            modes["number: gold is an exact multiple"] += 1
        elif g["type"] == "percent" and abs(a - b) <= .02:
            modes["percent: off by up to 0.02"] += 1
        elif (a < 0) != (b < 0):
            modes["number: wrong sign"] += 1
        else:
            q = abs(a - b) / max(abs(b), 1e-9)
            modes["number within 2%" if q < .02 else "number within 25%" if q < .25 else "number within 2x" if q < 1 else "number off by more"] += 1
    elif isinstance(a, list) and isinstance(b, list):
        x, y = [plain(v) for v in a], [plain(v) for v in b]
        modes["list: same items, wrong order" if sorted(x) == sorted(y) else f"list: length {'differs' if len(x) != len(y) else 'same'}, overlap {'>=half' if len(set(x) & set(y)) * 2 >= len(y) else '<half'}"] += 1
    elif g["type"] == "date":
        modes["date: other day"] += 1
    else:
        modes[f"{g['type']}: other value"] += 1
for k, v in modes.most_common():
    print(f"  {v:3d} {k}")
ag = Counter("share >= .9" if final_f[i][1] >= .9 else "share .75-.9" if final_f[i][1] >= .75 else "share < .75" for i in ids if after[i] == -1)
ar = Counter("share >= .9" if final_f[i][1] >= .9 else "share .75-.9" if final_f[i][1] >= .75 else "share < .75" for i in ids if after[i] == 1)
print("  wrong by agreement", dict(ag), "| right by agreement", dict(ar))

# E. seen against unseen shops, same model, prompt and samples
print("\n## E. Same adapter and prompt: shops it trained on against shops it never saw (4 thinking samples, vote)")
G = {**load("runs/sql_v3_gen30"), **load("runs/sql_v3_gen30b")}
U = load("runs/sql_v3_prompt632")
fin = lambda t: final({**t, "samples": t["samples"][2:6]}, 0)
seen = {i: S(i, fin(t)[0]) for i, t in G.items() if t["business_id"] not in hold_shops}
again = {i: S(i, fin(t)[0]) for i, t in G.items() if t["business_id"] in hold_shops}
unseen = {i: S(i, fin(t)[0]) for i, t in U.items()}
print(f"  seen shops: {len(seen)} q, {len({G[i]['business_id'] for i in seen})} shops, PA {pa(list(seen.values())):+.3f} {Counter(seen.values())}")
print(f"  unseen shops: {len(unseen)} q, PA {pa(list(unseen.values())):+.3f} {Counter(unseen.values())}")
for k in sorted({intent(Q[i]["question"]) for i in seen}):
    a = [v for i, v in seen.items() if intent(Q[i]["question"]) == k]
    b = [v for i, v in unseen.items() if intent(Q[i]["question"]) == k]
    print(f"    {k:14s} seen n={len(a):3d} PA {pa(a):+.2f} | unseen n={len(b):3d} PA {pa(b):+.2f}")
both = [i for i in again if i in unseen]
print(f"  run-to-run drift, same config on {len(both)} q: first {pa([again[i] for i in both]):+.3f}, second {pa([unseen[i] for i in both]):+.3f}, differing rows {sum(again[i] != unseen[i] for i in both)}")

# F. what the LoRA rows teach
print("\n## F. LoRA v2 training rows (data/sft/sql_rft_v2.jsonl)")
rows = rd("data/sft/sql_rft_v2.jsonl")
sql_of = lambda r: P.parse(r["reply"] if "</think>" not in r["reply"] else "<think>" + r["reply"])[1] or ""
kindr = lambda r: (lambda x: "views" if x and x <= fewshot.CANON else "raw" if x and not (x & fewshot.CANON) else "mixed" if x else "none")(fewshot.tables_of(sql_of(r)))
print("  rows", len(rows), "questions", len({r["id"] for r in rows}), "thinking", sum(r["thinking"] for r in rows), "with examples", sum("EXAMPLES FROM OTHER SHOPS" in r["user"] for r in rows), "| shop has views", sum("CANONICAL VIEWS" in r["user"] for r in rows), "has debt macros", sum("DEBTS (" in r["user"] for r in rows))
print("  source of the reply:", Counter(kindr(r) for r in rows).most_common(), "| where views are offered:", Counter(kindr(r) for r in rows if "CANONICAL VIEWS" in r["user"]).most_common())
print("  calls a debt macro:", sum(bool(P.DEBT_CALL.search(sql_of(r))) for r in rows))
print("  rows by intent:", Counter(intent(Q[r["id"]]["question"]) for r in rows).most_common())
print("  3000 train questions by intent:", Counter(intent(q["question"]) for q in Q.values()).most_common())
CONV = re.compile(r"quy_doi|quydoi|he_so|heso|sl_le|co_ban")
rq = [r for r in rows if intent(Q[r["id"]]["question"]) in ("ranking", "quantity") and re.search(r"(?i)sum\s*\(", sql_of(r))]
how = Counter()
for r in rq:
    s = sql_of(r)
    how["converted column or factor" if CONV.search(norm(s)) else "SUM(qty) on the view, raw at the time" if re.search(r'(?i)\bqty\b', s) else "other sum, e.g. raw quantity or amount"] += 1
print("  ranking and quantity rows that sum:", how.most_common())
template = lambda q: re.sub(r"\d+", "#", norm(q))
tc = Counter(template(Q[r["id"]]["question"]) for r in rows)
print(f"  distinct question templates in rows {len(tc)} | rows on the 20 commonest {sum(v for _, v in tc.most_common(20))} | templates seen once {sum(v == 1 for v in tc.values())}")
shops = Counter(Q[r["id"]]["business_id"] for r in rows)
print("  rows per shop: min", min(shops.values()), "median", st.median(shops.values()), "max", max(shops.values()), "of", len(shops), "shops")
print("  rows on holdout shops:", sum(Q[r["id"]]["business_id"] in hold_shops for r in rows))

# G. what each adapter changed, plain samples on the same questions
print("\n## G. Plain samples on the holdout: base, LoRA v1, LoRA v2")
B = {**load("runs/sql_q35_hold150"), **load("runs/sql_q35_base_rest450")}
V1 = {**load("runs/sql_q35_lora1_hold150"), **load("runs/sql_q35_lora1_rest450")}
rate = lambda T, xs: (lambda v: sum(v) / max(1, len(v)))([bool(s["pred"]) and S(i, s["pred"]) == 1 for i in xs for s in T[i]["samples"][:8]])
for name, T in (("base", B), ("LoRA v1", V1), ("LoRA v2", H)):
    allk = Counter(kind(s) for i in ids for s in T[i]["samples"][:8])
    print(f"  {name:8s} right {rate(T, ids):.3f} | no answer {sum(not s['pred'] for i in ids for s in T[i]['samples'][:8]) / (8 * len(ids)):.3f} | sources {[(k, round(v / (8 * len(ids)), 2)) for k, v in allk.most_common()]}")
print("  right rate by intent, base -> v1 -> v2:")
for k in sorted({intent(Q[i]["question"]) for i in ids}):
    g = [i for i in ids if intent(Q[i]["question"]) == k]
    print(f"    {k:14s} n={len(g):3d} {rate(B, g):.2f} -> {rate(V1, g):.2f} -> {rate(H, g):.2f}")

# H. the two scorers
d = Counter()
for i in ids:
    a, b = score_one({"id": i, **final_h[i][0]}, gold[i]), score2(final_h[i][0], gold[i])
    if a != b:
        d[gold[i]["type"], a, b] += 1
print("\n## H. score_one against evaluation.score on the same answers:", dict(d))
