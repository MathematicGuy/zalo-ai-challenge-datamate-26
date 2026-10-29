# DataMate — Zalo AI Challenge 2026

An agent that answers everyday Vietnamese questions over a small business's messy spreadsheets
(CSV / XLSX / SQLite), or abstains. Scored by Penalized Accuracy: +1 correct, 0 abstain, −1 wrong.

The LLM (Qwen3.5-9B) writes pandas code; a sandbox runs it; k sampled programs vote, and the agent
answers only when enough of them agree. Business rules are never hard-coded: every split ships its
own `rulebook.md`, which goes into the prompt verbatim, and the private test brings industries absent
from train.

## Status (2026-10-30)

Validation on the first 80 questions of `splits/val300.txt`:

| version | PA | precision when answering |
|---|---|---|
| v0 — raw tables, greedy | −0.075 | 0.44 |
| v1 — helpers, profiled columns, k=5 vote, τ=0.6 | +0.0625 | 0.65 |
| v2 — v1 + fixes from v1's logs | not measured yet | |

The v0→v1 gain is entirely the abstain gate; details in [docs/leaderboard.md](docs/leaderboard.md).

## Layout

```
src/datamate/
  io.py, tables.py, profile.py   read a split, detect header rows, cache tables
  context.py                     per-column profile shown to the model (README meaning, parser, date range, values)
  helpers.py                     rule-agnostic parsers injected into generated code (money, dates, fill-down)
  normalize.py                   chat-abbreviation expansion of questions
  retrieval/                     rule retrieval (lexical | dense | hybrid) and value linking
  prompts.py                     PromptSpec registry: v0, v1, v2
  llm.py                         ChatModel protocol, vLLM implementation
  sandbox.py                     isolated, parallel, time-limited execution of generated code
  answers.py                     code extraction, ANSWER validation, type guards
  select.py                      voting and abstention
  agent.py                       per-business pipeline: prompt → sample → execute → repair → vote
  predict.py                     CLI: submission.jsonl + per-question logs
  scoring.py, evaluate.py        PA metric from the Evaluation tab, run reports, τ sweep
notebooks/                       jupytext .py (outputs are not committed: they contain competition data)
docs/                            challenge spec, data notes, research → design decisions, results
splits/                          validation question ids, hand-made rule-relevance labels (train only)
tests/                           unit tests and a GPU-free end-to-end test with a scripted model
predict.sh                       Docker entrypoint
```

## Setup

```bash
uv venv ~/envs/datamate --python 3.12
uv pip install --python ~/envs/datamate/bin/python -e ".[llm,dev]"   # add ,retrieval for dense rule retrieval
export PATH=~/envs/datamate/bin:$PATH        # vLLM JIT-compiles kernels with ninja
```

Unzip `train.zip` and `public_test.zip` into `data/` (gitignored), giving `data/train/` and
`data/public_test/`. The first run caches parsed tables in `data/cache/` (a few minutes).

## Run

```bash
# validation (train questions, has answers)
python -m datamate.predict --data data/train --out runs/v2_val300 --ids splits/val300.txt --prompt v2
python -m datamate.evaluate runs/v2_val300 --ids splits/val300.txt      # PA, per type, τ sweep

# public test → upload runs/v2_public/submission.jsonl
python -m datamate.predict --data data/public_test --out runs/v2_public --prompt v2
```

Options: `--k` samples, `--temperature`, `--tau` abstain threshold, `--thinking`, `--rules-k` /
`--rule-retriever` (off by default). Engine settings are environment variables: `DATAMATE_GPU_UTIL`
(default 0.6), `DATAMATE_MAX_SEQS`, `DATAMATE_MAX_LEN`, `DATAMATE_TP`, `DATAMATE_QUANT`,
`DATAMATE_EXEC_PARALLEL`. A rerun into the same `--out` resumes where it stopped.

## Tests

```bash
python -m pytest -q        # needs data/train; no GPU
ruff check src tests notebooks
```

## Competition constraints this code respects

- Offline inference, ≤ 15B parameters in total (Qwen3.5-9B, Apache-2.0; embedders optional, 0.6B).
- Whole private test in ≤ 6 h on 2 × RTX 5090; answers are written per business, so a timeout keeps them.
- Per-question logs in `<out>/logs/` (Rule 8). Reproducibility: re-run PA within 0.03 of the upload.
- No training on test data; extra labels only on train (regulations §4.2).

Docs: [challenge spec](docs/challenge.md) · [data notes](docs/data-notes.md) ·
[research and design decisions](docs/research.md) · [results](docs/leaderboard.md)
