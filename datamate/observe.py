"""Read finished runs into tables for notebooks/observation.ipynb. No model is called.

A run dir is what scripts/run_compare.sh writes: submission.jsonl (merged), shard*/trace.jsonl (one record per question with
every sample), shard*/llm_calls.jsonl (one record per model request: tokens, latency, cost), shard*/run.json.

The reference of a split: its answers.jsonl when it has one (synth_private), else a silver key (public_test:
data/silver_public/submission.jsonl, a multi-agent Claude workflow that scored 0.998 on the leaderboard). Silver PA is a
diagnosis number, not a leaderboard score: a disagreement with silver is almost always a wrong answer, but not always.
"""
import json
from pathlib import Path

import pandas as pd

from .scoring import cluster_interval, metrics, score

REPO = Path(__file__).resolve().parents[1]


def read(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8-sig").splitlines() if l.strip()]


def reference(split, silver=None):
    """{id: gold} for a split, and whether it is real (answers.jsonl) or silver."""
    d = REPO / "data" / split
    if (d / "answers.jsonl").exists():
        return {r["id"]: r for r in read(d / "answers.jsonl")}, "answers"
    silver = Path(silver or REPO / "data/silver_public/submission.jsonl")
    return {r["id"]: r for r in read(silver)}, "silver"


def load_run(run_dir):
    """submission, trace, calls and run.json of one run (merged over its shards). None when the run has no submission yet."""
    run = Path(run_dir) if Path(run_dir).is_absolute() else REPO / run_dir
    sub_path = run / "submission.jsonl"
    if not sub_path.exists():
        return None
    shards = sorted(run.glob("shard*/")) or [run]
    trace = {r["id"]: r for s in shards if (s / "trace.jsonl").exists() for r in read(s / "trace.jsonl")}
    calls = [r for s in shards if (s / "llm_calls.jsonl").exists() for r in read(s / "llm_calls.jsonl")]
    info = [json.loads((s / "run.json").read_text()) for s in shards if (s / "run.json").exists()]
    return {"dir": run, "sub": {r["id"]: r for r in read(sub_path)}, "trace": trace, "calls": calls, "info": info}


def outcome(pred, gold):
    s = score(pred, gold)
    if s == 1:
        return "correct"
    if s == 0:
        return "abstain"
    if gold["type"] == "unanswerable":
        return "missed_unanswerable"
    if pred.get("type") == "unanswerable":
        return "false_unanswerable"
    return "wrong_type" if pred.get("type") != gold["type"] else "wrong"


def question_table(runs, silver=None):
    """One row per (model, split, question). runs: {model label: {split: run dir}}."""
    rows, loaded = [], {}
    for model, by_split in runs.items():
        for split, run_dir in by_split.items():
            r = load_run(run_dir)
            if r is None:
                continue
            loaded[model, split] = r
            gold, kind = reference(split, silver)
            qs = {q["id"]: q for q in read(REPO / "data" / split / "questions.jsonl")}
            for qid, g in gold.items():
                p = r["sub"].get(qid, {"type": "abstain"})
                t = r["trace"].get(qid, {})
                preds = [s.get("pred") for s in t.get("samples", [])]
                rows.append({"model": model, "split": split, "reference": kind, "id": qid, "business_id": qs.get(qid, {}).get("business_id"),
                             "question": qs.get(qid, {}).get("question"), "gold_type": g["type"], "pred_type": p.get("type"),
                             "score": score(p, g), "outcome": outcome(p, g), "pred": p, "gold": g,
                             "conf": t.get("conf"), "n_samples": len(preds), "n_thinking": sum("<think>" in (s.get("text") or "") for s in t.get("samples", [])),
                             "oracle": any(x and score(x, g) == 1 for x in preds), "elapsed_s": t.get("elapsed_seconds")})
    return pd.DataFrame(rows), loaded


def summary(df):
    """PA, precision, coverage and the per-shop bootstrap 95% interval per (split, model)."""
    out = []
    for (split, model), g in df.groupby(["split", "model"], sort=False):
        recs = g[["business_id", "score"]].to_dict("records")
        m = metrics(recs)
        lo, hi = cluster_interval(recs)
        out.append({"split": split, "model": model, "reference": g["reference"].iloc[0], "N": m["N"], "correct": m["correct"], "wrong": m["wrong"],
                    "abstain": m["abstain"], "pa": m["PA"], "lo": lo, "hi": hi, "precision": m["precision"], "coverage": m["coverage"],
                    "oracle": g["oracle"].mean()})
    return pd.DataFrame(out)


def call_table(loaded):
    """Tokens, latency, errors and cost per (model, split), from the request logs."""
    out = []
    for (model, split), r in loaded.items():
        c = pd.DataFrame(r["calls"])
        if c.empty:
            continue
        num = lambda k: pd.to_numeric(c[k], errors="coerce").fillna(0).sum() if k in c else 0
        secs = sum(i.get("seconds", 0) for i in r["info"])
        out.append({"model": model, "split": split, "requests": len(c), "errors": int(c["error"].notna().sum()) if "error" in c else 0,
                    "input_tokens": int(num("input_tokens") + num("prompt_tokens")), "cache_read_tokens": int(num("cache_read_tokens")),
                    "output_tokens": int(num("output_tokens")), "thought_tokens": int(num("thought_tokens")),
                    "cost_usd": round(float(num("cost_usd")), 2) if "cost_usd" in c else None, "wall_hours": round(secs / 3600, 2),
                    "median_request_s": float(pd.to_numeric(c.get("latency_s"), errors="coerce").median())})
    return pd.DataFrame(out)


def agreement(df, split):
    """Share of questions where two models give the same answer (abstains count as an answer 'abstain')."""
    from .voting import answer_key
    sub = df[df["split"] == split]
    keys = {m: dict(zip(g["id"], (answer_key(p) if p.get("type") != "abstain" else "ABSTAIN" for p in g["pred"]))) for m, g in sub.groupby("model", sort=False)}
    names = list(keys)
    return pd.DataFrame([[sum(keys[a][i] == keys[b].get(i) for i in keys[a]) / len(keys[a]) for b in names] for a in names], index=names, columns=names)
