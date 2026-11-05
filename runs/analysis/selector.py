"""Learned choice of the answer: for the two largest answer groups of a question, P(this group is right) from how the samples agree and how they got there; answer with the best group if P reaches a cut-off, else abstain.
Fitted on train labels only, no question or shop ids among the features. A logistic regression, so it ships as a JSON of weights and needs no extra package at inference.
python runs/analysis/selector.py eval --fit RUN... --test RUN...   leave-one-shop-out on the test runs, each fold also fitted on --fit, against the vote
python runs/analysis/selector.py fit RUN... --out sqlbase/selector_v1.json
python runs/analysis/selector.py apply RUN OUT --model sqlbase/selector_v1.json --data data/public_test [--cut C]"""
import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(R / "sqlbase"), str(R / "src"), str(R / "train")]
import predict as P
from check_submission import problem
from datamate.scoring import score_one
from ingest import norm

VIEWS, DEBT = re.compile(r"(?i)\bv_(sales|orders)(_all)?\b"), re.compile(r"debt_(at|new|repaid)\s*\(")
TYPES = ["number", "percent", "string", "date", "boolean", "list", "table", "unanswerable"]
INTENTS = [("delivery", r"dot_hang|dot_giao|giao_hang|xe_cho|xe_da_giao|xe_giao"), ("debt", r"(^|_)no(_|$)|cong_no|mua_chiu|ghi_no"), ("lot", r"(^|_)lo(_|$)|het_han|han_dung"),
           ("stock", r"(^|_)ton(_|$)|ton_kho|het_hang"), ("price", r"tang_gia|len_gia|gia_ban|doi_gia"), ("cost", r"(^|_)lai(_|$)|loi_nhuan|gia_von|nhap_hang|tien_nhap|nha_cung_cap|chi_phi"),
           ("rank", r"ban_chay|ban_nhieu|(^|_)top(_|$)|nhieu_nhat|cao_nhat|lon_nhat|it_nhat|hang_nao|mat_hang_nao"), ("percent", r"phan_tram|ti_le|ty_le|ti_trong|ty_trong|tang_truong"),
           ("customer", r"khach")]
NAMES = ["share", "share_plain", "share_think", "n_think", "other_share", "groups", "failed", "is_first", "view", "raw", "macro", "repaired", "any_macro", "zero", "negative", "qlen"] + TYPES + [n for n, _ in INTENTS]


def groups(t):
    """[(key, indices)] largest first, over samples with an answer."""
    g = defaultdict(list)
    for j, s in enumerate(t["samples"]):
        if s.get("pred"):
            g[P.answer_key(s["pred"])].append(j)
    return sorted(g.items(), key=lambda x: (-len(x[1]), x[1][0]))


def features(t, gs, rank):
    ss, idx = t["samples"], gs[rank][1]
    n, nt = len(ss), max(len(ss) - 8, 0)
    p = ss[idx[0]]["pred"]
    ans = p.get("answer")
    last = [ss[j]["text"].split("\n#REPAIR\n")[-1] for j in idx]
    other = len(gs[1 - rank][1]) / n if len(gs) > 1 and rank < 2 else 0
    num = isinstance(ans, (int, float)) and not isinstance(ans, bool)
    q = norm(t["question"])
    return [len(idx) / n, sum(j < 8 for j in idx) / min(n, 8), sum(j >= 8 for j in idx) / max(nt, 1), nt / 4, other, len(gs) / n, sum(not s.get("pred") for s in ss) / n, float(rank == 0),
            sum(bool(VIEWS.search(x)) for x in last) / len(idx), sum(not VIEWS.search(x) and not DEBT.search(x) for x in last) / len(idx), sum(bool(DEBT.search(x)) for x in last) / len(idx),
            sum("\n#REPAIR\n" in ss[j]["text"] for j in idx) / len(idx), float(any(DEBT.search(s["text"]) for s in ss)), float((num and ans == 0) or ans in ([], "", None)), float(num and ans < 0), len(t["question"]) / 100] \
        + [float(p.get("type") == x) for x in TYPES] + [float(bool(re.search(r, q))) for _, r in INTENTS]


def rows(runs, gold):
    """Question records: per question, its candidates with features, and each candidate's score."""
    out = {}
    for run in runs:
        for line in open(R / run / "trace.jsonl", encoding="utf-8"):
            t = json.loads(line)
            if t["final"].get("reason"):
                continue
            gs = groups(t)
            cand = []
            for rank in range(min(2, len(gs))):
                pred = t["samples"][gs[rank][1][0]]["pred"]
                cand.append((features(t, gs, rank), pred, score_one({"id": t["id"], **pred}, gold[t["id"]]) if gold else None))
            idx = P.debt_pool([s["text"] for s in t["samples"]], 8)
            a, c = P.vote([t["samples"][j]["pred"] for j in idx], len(idx))
            v = a if P.decide(a, c, 0.5) or (a.get("type") == "unanswerable" and c >= 0.5) else {"type": "abstain"}
            out[t["id"]] = {"shop": t["business_id"], "cand": cand, "vote": 0 if v["type"] == "abstain" or not gold else score_one({"id": t["id"], **v}, gold[t["id"]])}
    return list(out.values())


def fit(recs, c=1.0):
    from sklearn.linear_model import LogisticRegression
    X = np.array([f for r in recs for f, _, _ in r["cand"]])
    y = np.array([s == 1 for r in recs for _, _, s in r["cand"]], dtype=int)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    m = LogisticRegression(max_iter=3000, C=c).fit((X - mu) / sd, y)
    return {"names": NAMES, "coef": m.coef_[0].tolist(), "intercept": float(m.intercept_[0]), "mu": mu.tolist(), "sd": sd.tolist(), "n": len(y)}


def prob(model, f):
    z = (np.array(f) - np.array(model["mu"])) / np.array(model["sd"])
    return float(1 / (1 + np.exp(-(np.array(model["coef"]) @ z + model["intercept"]))))


def choose(model, r, cut):
    best = max(((prob(model, f), s) for f, _, s in r["cand"]), default=(0, 0), key=lambda x: x[0])
    return best[1] if best[0] >= cut else 0


def main():
    a = sys.argv[1]
    gold = {x["id"]: x for x in map(json.loads, open(R / "data/train/answers.jsonl"))}
    if a == "eval":
        p = argparse.ArgumentParser(); p.add_argument("--fit", nargs="*", default=[]); p.add_argument("--test", nargs="+")
        o = p.parse_args(sys.argv[2:])
        base, test = rows(o.fit, gold), rows(o.test, gold)
        shops = sorted({r["shop"] for r in test})
        test_ids = {id(r) for r in test}
        print(f"fit runs {len(base)} q, test {len(test)} q in {len(shops)} shops; right share of candidates in fit {np.mean([s == 1 for r in base for _, _, s in r['cand']]) if base else 0:.2f}")
        vote = {i: r["vote"] for i, r in enumerate(test)}
        print(f"vote: PA {sum(vote.values())/len(test):.3f} {sum(v == 1 for v in vote.values())}/{sum(v == -1 for v in vote.values())}/{sum(v == 0 for v in vote.values())}")
        for cut in (0.5, 0.55, 0.6, 0.65, 0.7, 0.75):
            s = {}
            for b in shops:
                tr = [r for r in base if r["shop"] != b] + [r for r in test if r["shop"] != b]
                m = fit(tr)
                for i, r in enumerate(test):
                    if r["shop"] == b:
                        s[i] = choose(m, r, cut)
            d = [s[i] - vote[i] for i in range(len(test))]
            boot = sorted(sum(random.Random(k).choices(d, k=len(d))) / len(d) for k in range(1000))
            vals = list(s.values())
            print(f"selector cut {cut}: PA {sum(vals)/len(vals):.3f} {vals.count(1)}/{vals.count(-1)}/{vals.count(0)}, delta {sum(d)/len(d):+.3f} CI [{boot[25]:+.3f}, {boot[974]:+.3f}], moved {sum(x != 0 for x in d)}")
        m = fit(base + test)
        print("largest weights:", [(n, round(c, 2)) for n, c in sorted(zip(NAMES, m["coef"]), key=lambda x: -abs(x[1]))[:10]])
    elif a == "fit":
        p = argparse.ArgumentParser(); p.add_argument("runs", nargs="+"); p.add_argument("--out"); o = p.parse_args(sys.argv[2:])
        m = fit(rows(o.runs, gold))
        (R / o.out).write_text(json.dumps(m, indent=1))
        print(f"fitted on {m['n']} candidates -> {o.out}")
    elif a == "apply":
        p = argparse.ArgumentParser(); p.add_argument("run"); p.add_argument("out"); p.add_argument("--model"); p.add_argument("--data", default="data/public_test"); p.add_argument("--cut", type=float, default=0.5)
        o = p.parse_args(sys.argv[2:])
        m = json.load(open(R / o.model))
        qs = [json.loads(x) for x in open(R / o.data / "questions.jsonl", encoding="utf-8-sig") if x.strip()]
        res, c = {}, Counter()
        for line in open(R / o.run / "trace.jsonl", encoding="utf-8"):
            t = json.loads(line)
            if t["final"].get("reason"):
                res[t["id"]] = t["final"]; c["rule"] += 1
                continue
            gs = groups(t)
            cand = [(prob(m, features(t, gs, k)), t["samples"][gs[k][1][0]]["pred"]) for k in range(min(2, len(gs)))]
            best = max(cand, key=lambda x: x[0], default=(0, None))
            r = best[1] if best[0] >= o.cut else {"type": "abstain"}
            res[t["id"]] = {"type": "abstain"} if problem({"id": t["id"], **r}) else r
            c["answered" if res[t["id"]]["type"] != "abstain" else "abstain"] += 1
        (R / o.out).mkdir(parents=True, exist_ok=True)
        P.write_submission(R / o.out / "submission.jsonl", qs, res)
        print(dict(c), "->", o.out)


if __name__ == "__main__":
    main()
