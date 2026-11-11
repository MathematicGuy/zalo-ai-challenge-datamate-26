"""Writes notebooks/observation.ipynb (run: .venv/bin/python notebooks/build_observation.py). Edit here, not in the notebook."""
from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
md("""# Observation · three models on the same pipeline

This notebook reads finished runs and compares them. It never calls a model, so it runs on any machine in seconds.

**What is compared.** The DataMate pipeline is fixed: ingest the shop, draft canonical views and debt macros, then K = 8 SQL samples per question, extra thinking samples when agreement is low, repairs, and a vote. Only the model behind it changes:

| Model | How it runs | Notes |
|---|---|---|
| **Qwen3.5-9B + SQL LoRA** | local, the tuned vLLM server in `inference/qwen35-9b` (fp8 weights, fp8 KV cache, one 16 GB GPU) | the submitted model; context 32k instead of 49k on this GPU |
| **Claude Haiku 5.5** | Anthropic API, prompt caching on the shop context | no LoRA, no prefill (emulated), no sampling temperature (API default) |
| **Gemini 3.5 Flash Lite** | Google API, rotating over every `GEMINI_API_KEY*` | no LoRA, no prefill (emulated) |

Every model drafts its own views and debt layer: the comparison is end to end.

**References.**
- `synth_private` (498 questions, 15 shops in new and familiar industries) has real answers: its PA is exact.
- `public_test` (1000 questions) has no answers. It is scored against the **silver key**: answers from a multi-agent Claude workflow that scored PA 0.998 on the leaderboard. A disagreement with silver is almost always a wrong answer, but a few silver answers may be wrong. Read public numbers as "silver PA", a diagnosis number.

**Penalized Accuracy (PA)** = (right - wrong) / N. Abstaining scores 0, so a model that knows when it doesn't know beats one that guesses.

How to read it:
1. Overview tiles
2. PA with confidence intervals
3. By answer type
4. By shop
5. Outcome mix
6. Cost, tokens and time
7. Sampling headroom: oracle and agreement
8. Do the models agree?
9. Inspect disagreements"""),
code("""# Setup: the repo on sys.path, charts inline
import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))

import pandas as pd
from IPython.display import display

from datamate import observe
from datamate import obs_charts as ch

pd.set_option("display.max_colwidth", 90)
pd.set_option("display.width", 200)"""),
code("""# Runs to compare: {model label: {split: run dir}}. A run that does not exist yet is skipped.
RUNS = {
    "Qwen3.5-9B + LoRA": {"public_test": "runs/qwen35_public", "synth_private": "runs/qwen35_private"},
    "Claude Haiku 5.5": {"public_test": "runs/haiku55_public", "synth_private": "runs/haiku55_private"},
    "Gemini 3.5 Flash Lite": {"public_test": "runs/gemini35fl_public", "synth_private": "runs/gemini35fl_private"},
}
SILVER = "data/silver_public/submission.jsonl"   # reference for public_test (no answers.jsonl)

df, loaded = observe.question_table(RUNS, silver=ROOT / SILVER)
summ = observe.summary(df)
print(f"{len(loaded)} runs loaded:", ", ".join(f"{m} / {s}" for m, s in loaded))"""),
md("""## 1 · Overview

One tile per run: PA, then how it splits into right, wrong and abstain. **Precision** is right / answered. **Coverage** is answered / N."""),
code("""for split in summ["split"].unique():
    s = summ[summ["split"] == split]
    ref = "real answers" if s["reference"].iloc[0] == "answers" else "silver key"
    print(f"\\n{split} ({s['N'].iloc[0]} questions, scored against {ref})")
    ch.stat_tiles([(f"{r.pa:+.3f}", r.model, f"{r.correct} right · {r.wrong} wrong · {r.abstain} abstain · precision {r.precision:.0%}") for r in s.itertuples()])"""),
code("""summ.round(3)"""),
md("""## 2 · PA with 95% intervals

The interval resamples **whole shops** (questions of one shop share the same data quirks, so they are not independent). Two models whose intervals overlap a lot are not clearly different on this sample."""),
code("""ch.model_pa_bars(summ, "PA by model and split", "bars = 95% interval, shop-level bootstrap")"""),
md("""## 3 · By answer type

Where each model wins or loses. Numbers dominate both sets, while lists, tables and dates are rarer but harder. `unanswerable` rewards a model that dares to say no."""),
code("""for split in df["split"].unique():
    sub = df[df["split"] == split]
    t = sub.pivot_table(index="gold_type", columns="model", values="score", aggfunc="mean")
    t = t.loc[sub["gold_type"].value_counts().index]  # most common type first
    ch.grouped_bars(t, f"{split}: PA by reference answer type")
    display(sub.groupby(["gold_type", "model"])["score"].agg(n="size", PA="mean").unstack().round(3))"""),
md("""## 4 · By shop

Each shop is its own messy data world (headers, units, dates, debts). A model that is strong on average can still fail a whole shop when its view draft for that shop is wrong."""),
code("""for split in df["split"].unique():
    t = df[df["split"] == split].pivot_table(index="business_id", columns="model", values="score", aggfunc="mean")
    ch.grouped_bars(t.sort_index(), f"{split}: PA by shop")"""),
md("""## 5 · Outcome mix

- **Wrong value**: right answer type, wrong value.
- **Wrong answer type**: e.g. a number where a percent was expected.
- **Should be unanswerable**: the reference says the data cannot answer, but the model answered.
- **Unanswerable by mistake**: the model said unanswerable when the data could answer.
- **Abstain**: the vote was too split (below 50%), so the pipeline left it blank, scoring 0."""),
code("""counts = df.assign(run=df["split"] + " · " + df["model"]).groupby(["run", "outcome"]).size().unstack(fill_value=0)
ch.outcome_stack(counts, "Outcome mix per run")
counts"""),
md("""## 6 · Cost, tokens and time

From every request the engines logged (`llm_calls.jsonl`).
- **Claude**: cost uses Haiku 5.5 list prices ($0.10 / $0.50 per MTok, cache reads $0.01).
- **Gemini**: tokens only.
- **Qwen**: local; only wall time matters (one RTX 5060 Ti 16 GB).

`seconds per question` is the shop's batch time divided by its questions; shared view drafting is excluded."""),
code("""calls = observe.call_table(loaded)
display(calls)
sec = {f"{s} · {m}": g["elapsed_s"].dropna() for (s, m), g in df.groupby(["split", "model"], sort=False)}
ch.latency_box(sec, "Seconds per question (shop batch time / questions)")"""),
md("""## 7 · Sampling headroom: oracle and agreement

- **Oracle** = share of questions where *at least one* sample was right: the ceiling a perfect answer picker could reach. The gap between oracle and the right-answer rate is what better voting could still win.
- **Vote share** (`conf`) = share of samples behind the voted answer. A calibrated model is right when it agrees with itself."""),
code("""o = df.groupby(["split", "model"], sort=False).agg(right=("score", lambda s: (s == 1).mean()), oracle=("oracle", "mean"), mean_samples=("n_samples", "mean"),
                                                  thinking_share=("n_thinking", lambda s: (s > 0).mean())).round(3)
display(o)
df["conf_bin"] = pd.cut(df["conf"].astype(float), [0, .25, .5, .75, .99, 1.0], include_lowest=True)
df[df["score"] != 0].pivot_table(index="conf_bin", columns=["split", "model"], values="score", aggfunc=lambda s: (s == 1).mean(), observed=False).round(2)"""),
md("""## 8 · Do the models agree?

Share of questions where two models give the same answer (keys follow the scorer: numbers to 2 decimals, text case-insensitive; `abstain` is an answer here). High agreement with different PA means the disagreements are where one model is right."""),
code("""for split in df["split"].unique():
    ch.heatmap(observe.agreement(df, split), f"{split}: answer agreement between models")"""),
code("""# Paired difference: for each pair, how many questions each model alone gets right (same questions, same reference)
for split in df["split"].unique():
    w = df[df["split"] == split].pivot_table(index="id", columns="model", values="score")
    names = list(w.columns)
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d = w[a] - w[b]
            rows.append({"A": a, "B": b, "ΔPA (A-B)": round(d.mean(), 3), "A right, B not": int(((w[a] == 1) & (w[b] != 1)).sum()),
                         "B right, A not": int(((w[b] == 1) & (w[a] != 1)).sum())})
    print(split); display(pd.DataFrame(rows))"""),
md("""## 9 · Inspect disagreements

Questions where the models disagree and exactly one is right, then one question in full: every model's final answer next to the reference, and the samples behind each vote. Change `SPLIT` / `PICK` to browse."""),
code("""SPLIT = "synth_private"
w = df[df["split"] == SPLIT].pivot_table(index="id", columns="model", values="score")
one_right = w[(w == 1).sum(axis=1) == 1]
print(f"{len(one_right)} questions where exactly one model is right")
q = df[df["split"] == SPLIT].drop_duplicates("id").set_index("id")
one_right.join(q[["business_id", "gold_type", "question"]]).head(15)"""),
code("""PICK = one_right.index[0] if len(one_right) else w.index[0]
rows = df[(df["split"] == SPLIT) & (df["id"] == PICK)]
print(rows["question"].iloc[0], "\\nreference:", rows["gold"].iloc[0])
display(rows[["model", "outcome", "pred", "conf", "n_samples"]])
for (model, split), r in loaded.items():
    t = r["trace"].get(PICK)
    if split == SPLIT and t:
        print(f"\\n── {model}: vote {t.get('vote')} (share {t.get('conf')})")
        for s in t["samples"][:12]:
            print("  ", "think" if "<think>" in (s.get("text") or "") else "plain", "→", s.get("pred") or ("error: " + str(s.get("error"))[:120]))"""),
md("""## Takeaways

Fill in after reading the numbers above:
- Which model has the best PA on real answers (`synth_private`), and is the gap outside the shop-level interval?
- Does the public silver ranking agree with the private one?
- Where is the remaining headroom: oracle minus right (better voting), or questions no sample gets right (better model or prompts)?"""),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
nbf.write(nb, Path(__file__).with_name("observation.ipynb"))
print("wrote notebooks/observation.ipynb")
