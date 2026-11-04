# DataMate SQL pipeline

Answers Vietnamese business questions over each shop's CSV/XLSX/SQLite exports (Zalo AI Challenge 2026). Scoring is +1 correct, −1 wrong, 0 abstain, so the system answers only when its samples agree. Setup steps are in the [root README](../README.md).

No answers, business ids or question templates are hard-coded; inference never reads `answers.jsonl`. Total parameters at inference: 9.70B (`model_manifest.json`).

## Method

1. **Load** (`ingest.py`): every sheet and table becomes a typed DuckDB table with its original column names. With `--canon-names`, names that differ only in case, spaces or missing accents are written one way across the shop's tables, so an item or a customer is one value in the raw tables, the catalog joins and the views.
2. **Views per shop** (`llmviews.py`, base weights, thinking): 8 drafts of one SELECT mapping the shop to `v_sales`, `v_orders` and their `_all` versions, data tests, one repair turn, most common fingerprint. If too few drafts pass the catalog test, the most common draft that only fails that test is used. `qty` is in base units; if price per unit still differs across units of an item (`unit_spread` above 2), the factors are read from the prices (`qty_by_price`).
3. **Debt layer per shop** (`debtview.py`): the same recipe gives the macros `debt_at`, `debt_new` and `debt_repaid`.
4. **Samples** (`predict.py`, LoRA): 8 non-thinking SQL samples, half with similar examples from other shops (`fewshot.py`), executed on DuckDB with up to 2 repair turns. Questions under 0.75 agreement get 4 thinking samples.
5. **Vote**: answer at a share of 0.5 or more, abstain on 0 or empty results. `--debt-route` votes among macro-using samples when most thinking samples use a macro. `--answer-unanswerable` answers when that is the majority. A period wholly outside the shop's dates is unanswerable.

## Run

```bash
bash sqlbase/run_public.sh runs/my_run            # public test, saved views, about 2 h on one 24 GB GPU
bash runs/analysis/run_v2_public.sh runs/old_run  # the older 0.491 setup
python3 train/check_submission.py runs/my_run/submission.jsonl
```

`predict.py --help` lists every flag. Outputs: `submission.jsonl`, `trace.jsonl` (all samples, SQL, errors, votes), `run.json`, `logs/`.

`attention_backend` is `TRITON_ATTN` because FlashInfer JIT needs nvcc 12.9 or newer on sm120 GPUs.

## Docker

Build from the repo root with `docker build -f sqlbase/Dockerfile -t datamate .` The image holds the base model and LoRA v2. `predict.sh` reads `/data`, writes `/result/submission.jsonl`, runs one process per GPU over the shops and merges them, with a 5h50m limit. It drafts views itself, so it needs no saved config. Override with `MODEL_DIR`, `LORA_DIR`, `GPU_MEM`, `ENGINE`, `NUM_GPUS` and `TIMEOUT_SECONDS`.

## Training

Environment: `train/requirements.txt`, in a separate venv from inference.

1. Generate samples on train with `--save-prompts`.
2. `runs/analysis/build_sft_v3.py` keeps samples that reach the gold answer, on the best route per question: debt macros, then views, then raw tables.
3. `train/train_lora.py` trains QLoRA, r=16, continuing from the previous adapter.

`runs/analysis/chain_v3.sh` is the full recipe for LoRA v3. It also runs Qwen3.8-27B on the train questions no 9B sample solved; that model is used for training data only, never at inference or on test data.

## Model

Qwen/Qwen3.5-9B revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a` (9.65B params, Apache-2.0), plus a 43.3M-parameter LoRA adapter. Seed 0.
