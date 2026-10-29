# DataMate — Zalo AI Challenge 2026

An agent that answers everyday Vietnamese questions over a small business's messy spreadsheets
(CSV / XLSX / SQLite), or abstains. Scored by Penalized Accuracy: +1 correct, 0 abstain, −1 wrong.

The tables are normalized once at load time (parsed date / money columns, fill-down, combined
monthly sheets). Qwen3.5-9B writes pandas code over them, a sandbox runs it, K sampled programs vote,
and the agent answers only when enough of them agree. Business rules are never hard-coded: each split
ships its own `rulebook.md`, which goes into the prompt verbatim.

## Layout

```
notebooks/
  01_data.ipynb       data layer (read, header detection, normalization, table description) + EDA + data cards
  02_agent.ipynb      the agent: prompts v2/v3, retrieval, LLM, sandboxed execution, voting, submission
  03_evaluate.ipynb   scores, τ sweep, error reading, rule-retrieval study, normalization check
src/datamate/         the parts that must be importable files
  helpers.py          parsers injected into the generated code (money, dates, fill-down, lunar dates)
  sandbox.py          spawned worker that runs generated code with a timeout
  scoring.py          the PA metric from the Evaluation tab
  nblib.py            run_lib_cells: load a notebook's `lib` cells into another notebook
run.sh                run 02_agent headless on a split, one shard per GPU
predict.sh            Docker entrypoint: /data -> /result
docs/                 challenge spec, data notes, research → design decisions, results
splits/               validation ids; hand-made rule-relevance labels (train only)
tests/                pytest, no GPU (scripted model for the end-to-end test)
```

Cells tagged `lib` in 01 and 02 are the code; the other cells explore or run. 02 and 03 load the
`lib` cells of the notebooks before them, so editing a function in 01 changes it everywhere.

## Setup (server)

Needs CUDA ≥ 12.8 (Blackwell), Python 3.12.

```bash
git clone https://github.com/hugebenevolence/zalo-ai-challenge-datamate-26 && cd zalo-ai-challenge-datamate-26
uv venv .venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[llm,dev]"          # add ,retrieval only for dense rule retrieval
hf download Qwen/Qwen3.5-9B             # ~19 GB, once
```

Put the released zips in `data/` and unzip them there: `data/train/`, `data/public_test/`
(`data/` is gitignored). The first run caches normalized tables in `data/cache/` (a few minutes).

## Run

```bash
pytest -q                                                    # CPU only, ~15 s: everything but the LLM

# 1. smoke test on the GPUs: 6 questions, 1 sample each
./run.sh train runs/smoke -p IDS ../splits/smoke6.txt -p K 1

# 2. validation: v3, and v2 on the same questions for comparison
./run.sh train runs/v3_val300 -p IDS ../splits/val300.txt
./run.sh train runs/v2_val300 -p IDS ../splits/val300.txt -p PROMPT v2

# 3. public test -> upload runs/v3_public/submission.jsonl
./run.sh public_test runs/v3_public
```

`run.sh` starts one process per visible GPU (`DATAMATE_NGPU` to override), each on a share of the
shops, and merges `submission.jsonl` + `logs/` when all finish. A rerun into the same folder resumes.
Progress is in `runs/<run>/shard<i>.txt`. Parameters go through papermill: `-p K 5 -p TAU 0.6
-p TEMPERATURE 0.7 -p THINKING True -p PROMPT v2` (see the first cell of `02_agent.ipynb`).
Engine settings are environment variables; on GPUs nobody else uses:

```bash
export DATAMATE_GPU_UTIL=0.9 DATAMATE_MAX_SEQS=48
```

Then open `03_evaluate.ipynb`, set `RUNS = ["v3_val300", "v2_val300"]` and run it: PA per version on
the same questions, τ sweep, wrong answers with their code and output.

**Send back** `runs/<run>/` (submission, logs, shard*.txt). They contain competition data: share them
inside the team only.

## Status (2026-10-30)

| version | validation | notes |
|---|---|---|
| v0 | PA −0.075 on 80 q | raw tables, greedy |
| v1 | PA +0.0625 on 80 q | gain is the abstain gate only (τ=0 precision 0.24) |
| v2 | PA −0.233 on 30 q | the model still parsed raw dates itself → NaT → answered 0 |
| v3 | **not run yet** | normalized tables; checked against 71 train answers without a GPU |

Details: [docs/leaderboard.md](docs/leaderboard.md). The dev machine's GPU driver is down, so v3's
first real numbers come from the server runs above.

## Before committing

Notebook outputs contain competition data (regulations §4.3); a test fails if a committed notebook
has outputs. Clear them first:

```bash
jupyter nbconvert --clear-output --inplace notebooks/*.ipynb && pytest -q && ruff check src tests notebooks
```

## Competition constraints this code respects

- Offline inference, ≤ 15B parameters in total (Qwen3.5-9B, Apache-2.0; optional embedders 0.6B).
- Whole private test in ≤ 6 h on 2 × RTX 5090; answers are written per shop, so a timeout keeps them.
- Per-question logs in `<out>/logs/` (Rule 8); re-run PA must stay within 0.03 of the upload.
- No training on test data; extra labels only on train (regulations §4.2).

Docs: [challenge spec](docs/challenge.md) · [data notes](docs/data-notes.md) ·
[research and design decisions](docs/research.md) · [results](docs/leaderboard.md)
