"""Score a run against train answers: python -m datamate.evaluate runs/<run> [--ids splits/val300.txt]

Also re-decides every question from its logged samples at several abstain thresholds (no new LLM
calls). Tune the threshold here on validation, never on the public test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .answers import postprocess
from .io import load_answers, load_questions, read_jsonl
from .scoring import penalized_accuracy, score_one
from .select import vote


def load_refs(split: str, ids: set[str] | None) -> dict[str, dict]:
    return {r["id"]: r for r in load_answers(split).to_dict("records") if ids is None or r["id"] in ids}


def load_logs(run_dir: Path) -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in (run_dir / "logs").glob("*.json")}


def evaluate(run_dir: Path, ids: set[str] | None = None, split: str = "train") -> tuple[dict, pd.DataFrame]:
    refs = load_refs(split, ids)
    preds = {r["id"]: r for r in read_jsonl(run_dir / "submission.jsonl")}
    res = penalized_accuracy(preds, refs)
    q = load_questions(split).set_index("id")
    df = pd.DataFrame([{"id": k, "score": v, "ref_type": refs[k]["type"],
                        "pred_type": preds.get(k, {}).get("type", "missing"),
                        "business_id": q.loc[k, "business_id"], "question": q.loc[k, "question"]}
                       for k, v in res.pop("per_question").items()])
    return res, df


def revote(lg: dict, k: int, post: bool = True) -> dict:
    """A question's vote recomputed from the per-sample answers in its log (logged from v2 on),
    optionally passing each through answers.postprocess."""
    answers = [x.get("answer") for x in lg["samples"] if not x.get("repair")]
    if post:
        answers = [postprocess(lg["question"], a) for a in answers]
    return vote(answers, k)


def _logged_vote(lg: dict | None, k: int, per_sample: bool, post: bool) -> dict:
    if lg is None:
        return {"answer": None, "support": 0.0}
    if per_sample:
        return revote(lg, k, post)
    top = lg["clusters"][0]["answer"] if lg.get("clusters") else None   # runs before v2: winning cluster only
    return {"answer": top, "support": lg.get("support", 0.0)}


def sweep_tau(run_dir: Path, ids: set[str] | None = None, taus=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
              split: str = "train", post: bool = True) -> pd.DataFrame:
    refs, logs = load_refs(split, ids), load_logs(run_dir)
    k = json.loads((run_dir / "config.json").read_text())["k"]
    per_sample = any("answer" in x for lg in logs.values() for x in lg["samples"])
    votes = {qid: _logged_vote(logs.get(qid), k, per_sample, post) for qid in refs}
    rows = []
    for tau in taus:
        s = pd.Series([score_one(v["answer"] if v["answer"] is not None and v["support"] >= tau else None, refs[qid])
                       for qid, v in votes.items()])
        answered = int((s != 0).sum())
        rows.append({"tau": tau, "PA": round(s.mean(), 4), "answered": answered,
                     "precision": round((s == 1).sum() / answered, 3) if answered else float("nan")})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, by: str) -> pd.DataFrame:
    g = df.groupby(by).score
    return pd.DataFrame({"n": g.size(), "PA": g.mean().round(3), "correct": g.apply(lambda s: (s == 1).sum()),
                         "wrong": g.apply(lambda s: (s == -1).sum()), "abstain": g.apply(lambda s: (s == 0).sum())})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--ids", type=Path)
    a = ap.parse_args()
    ids = set(a.ids.read_text().split()) if a.ids else None
    res, df = evaluate(a.run_dir, ids)
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()}))
    print(summarize(df, "ref_type").to_string())
    print(pd.crosstab(df.ref_type, df.pred_type).to_string())
    logs = load_logs(a.run_dir)
    if logs and "clusters" in next(iter(logs.values())):
        print(sweep_tau(a.run_dir, ids).to_string(index=False))


if __name__ == "__main__":
    main()
