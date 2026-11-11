#!/usr/bin/env bash
# One model on one split with the submitted pipeline flags, for model comparisons (notebooks/observation.ipynb reads the runs).
#
#   bash scripts/run_compare.sh BACKEND SPLIT [OUT] [extra predict args...]
#     BACKEND  openai  Qwen3.5-9B + SQL LoRA on the tuned vLLM server (start it first: see inference/qwen35-9b/README.md)
#              claude  Claude Haiku 5.5 (ANTHROPIC_API_KEY in the env or ./env)
#              gemini  Gemini 3.5 Flash Lite (GEMINI_API_KEY_1..N in the env or ./env, rotated)
#     SPLIT    a folder under data/ (public_test, synth_private)
#
# Every layer (views, debt macros) is drafted again by the model under test (no configs/public_v9 reuse). On the 16 GB server the
# context is 32768 (submission: 49152) and the thinking budget 12288 (submission: 16384). MODEL overrides the model id.
set -eu
cd "$(dirname "$0")/.."
source scripts/flags.sh
BACKEND=$1; SPLIT=$2; shift 2
OUT=${1:-runs/${BACKEND}_${SPLIT}}; [ $# -gt 0 ] && shift
PY=${PYTHON:-.venv/bin/python}

# the submitted flags without the in-process engine settings (--quant/--gpu-mem live on the server)
FLAGS=(--k 8 --think-k 4 --notes --max-tokens 2048 --debt-views --debt-route --canon-names --answer-unanswerable)
case "$BACKEND" in
  openai) MODEL=${MODEL:-qwen3.5-9b}; FLAGS+=(--context 32768 --think-budget 12288 --lora adapters/sql_rft_v3 --lora-thinking) ;;
  claude) MODEL=${MODEL:-claude-haiku-5-5}; FLAGS+=(--context 200000 --think-budget 12288) ;;
  gemini) MODEL=${MODEL:-gemini-3.5-flash-lite}; FLAGS+=(--context 200000 --think-budget 12288) ;;
  *) echo "unknown backend $BACKEND" >&2; exit 2 ;;
esac

mkdir -p "$OUT"
"$PY" -m datamate.predict --backend "$BACKEND" --model "$MODEL" --data "data/$SPLIT" --output "$OUT/shard0" \
  --fewshot assets/fewshot_bank.jsonl --save-prompts "${FLAGS[@]}" "$@" 2>&1 | tee "$OUT/shard0.log"
"$PY" -m datamate.submission merge "$OUT" --questions "data/$SPLIT/questions.jsonl"
"$PY" -m datamate.submission check "$OUT/submission.jsonl" --questions "data/$SPLIT/questions.jsonl"
