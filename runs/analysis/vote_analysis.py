"""Where the final decision loses points, on labeled runs, with the public submission policy (apply_review.py: debt pool, vote at 0.5, abstain on
0/empty, answer unanswerable at share >= .5): python runs/analysis/vote_analysis.py RUN [RUN ...]
Per question: is a right answer among the samples, is it the voted group, its share, nil (0/empty), unanswerable; then what the policy did with it."""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path
R = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(R / "sqlbase"), str(R / "src")]
import predict as P
from datamate.scoring import score_one

gold = {x["id"]: x for x in map(json.loads, open(R / "data/train/answers.jsonl"))}
rows = []
for run in sys.argv[1:]:
    for line in open(R / run / "trace.jsonl", encoding="utf-8"):
        t = json.loads(line); g = gold[t["id"]]
        ok = lambda p: bool(p) and p.get("type") not in ("abstain",) and score_one({"id": t["id"], **p}, g) == 1
        idx = P.debt_pool([s["text"] for s in t["samples"]], 8)
        preds = [t["samples"][j]["pred"] for j in idx]
        ans, conf = P.vote(preds, len(preds))
        if t["final"].get("reason"):
            final = t["final"]
        else:
            final = ans if P.decide(ans, conf, 0.5) else {"type": "abstain"}
            if ans.get("type") == "unanswerable" and conf >= 0.5:
                final = ans
        groups = defaultdict(list)
        for p in preds:
            if p:
                groups[P.answer_key(p)].append(p)
        right_share = max((len(v) / len(preds) for v in groups.values() if any(ok(p) for p in v)), default=0)
        nil = ans.get("answer") in (0, [], None) and ans.get("type") not in ("boolean", "unanswerable", "abstain")
        top_ok = ok(ans)
        if ok(final):
            o = "right"
        elif final["type"] != "abstain":
            o = "WRONG (-1)" if not top_ok else "?"
        elif top_ok and nil:
            o = "abstain: right top group is 0/empty"
        elif top_ok and conf < 0.5:
            o = "abstain: right top group below 0.5"
        elif top_ok:
            o = "abstain: other"
        elif right_share > 0:
            o = "abstain: right answer in a minority group"
        elif not any(preds):
            o = "abstain: no sample executed"
        else:
            o = "abstain: no right sample"
        think = sum(1 for j in idx if j >= 8)
        rows.append(dict(id=t["id"], shop=t["business_id"], gtype=g["type"], o=o, conf=conf, right_share=right_share, top_ok=top_ok, nil=nil,
                         atype=ans.get("type"), n=len(preds), think=think, final=final, ans=ans, gold=g, q=t["question"]))
N = len(rows)
c = Counter(r["o"] for r in rows)
pa = (c["right"] - c["WRONG (-1)"]) / N
print(f"# Decision analysis, {N} questions, public policy: PA {pa:.3f}")
for k, v in c.most_common():
    print(f"  {k:<45} {v:>5}  {100 * v / N:.1f}%")
print("\n## outcome by gold type")
by = defaultdict(Counter)
for r in rows:
    by[r["gtype"]][r["o"]] += 1
cols = [k for k, _ in c.most_common()]
print(f"{'':<14}" + "".join(f"{k[:24]:>26}" for k in cols))
for gt, cc in sorted(by.items(), key=lambda x: -sum(x[1].values())):
    print(f"{gt:<14}" + "".join(f"{cc[k]:>26}" for k in cols))
print("\n## wrong answers given (-1): voted share and type")
w = [r for r in rows if r["o"] == "WRONG (-1)"]
print("  share:", Counter(round(r["conf"], 1) for r in w).most_common(), "| type:", Counter(r["atype"] for r in w).most_common(), "| gold type:", Counter(r["gtype"] for r in w).most_common())
print("\n## threshold sweep (same rules otherwise): threshold -> right, wrong, abstain, PA")
for th in (0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7):
    rt = wr = 0
    for r in rows:
        a = r["ans"]
        if r["final"].get("reason"):
            f = r["final"]
        else:
            f = a if P.decide(a, r["conf"], th) else {"type": "abstain"}
            if a.get("type") == "unanswerable" and r["conf"] >= th:
                f = a
        if f["type"] == "abstain":
            continue
        good = score_one({"id": r["id"], **f}, r["gold"]) == 1
        rt += good; wr += not good
    print(f"  {th:.2f}: {rt} {wr} {N - rt - wr}  PA {(rt - wr) / N:.3f}")
print("\n## 0/empty voted answers: how often the 0/empty group is right")
nil = [r for r in rows if r["nil"] and r["conf"] >= 0.5]
print(f"  {len(nil)} questions vote 0/empty at share >= .5; right {sum(r['top_ok'] for r in nil)}; gold 0/empty in train: "
      f"{sum(1 for g in gold.values() if g['answer'] in (0, []) and g['type'] not in ('boolean',))}")
print("\n## abstained with the right answer as the top group below 0.5, by share")
lo = [r for r in rows if r["o"] == "abstain: right top group below 0.5"]
print("  ", Counter(round(r["conf"], 2) for r in lo).most_common(12))
print("  same shares among WRONG top groups (would turn -1 if answered):", Counter(round(r["conf"], 2) for r in rows if not r["top_ok"] and r["ans"].get("type") != "abstain" and r["conf"] < 0.5 and not r["nil"]).most_common(12))
print("\n## unanswerable gold: what the vote said")
un = [r for r in rows if r["gtype"] == "unanswerable"]
print("  voted type:", Counter(r["atype"] for r in un).most_common(), "| share of unanswerable among samples:",
      Counter(round(sum(1 for _ in []) or 0, 1) for r in un).most_common(0))
for r in un:
    if r["o"] != "right":
        print(f"   {r['shop']} | {r['q'][:80]} | vote {r['atype']} {json.dumps(r['ans'].get('answer'), ensure_ascii=False)[:30]} at {r['conf']:.2f} | {r['o']}")
print("\n## answered unanswerable at share >= .5 but gold has an answer (-1)")
for r in rows:
    if r["o"] == "WRONG (-1)" and r["atype"] == "unanswerable":
        print(f"   {r['shop']} | {r['q'][:80]} | gold {r['gtype']} {json.dumps(r['gold']['answer'], ensure_ascii=False)[:40]} | share {r['conf']:.2f}")
with open(R / "runs/debug/vote_rows.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps({k: v for k, v in r.items() if k not in ("gold",)}, ensure_ascii=False) + "\n")
