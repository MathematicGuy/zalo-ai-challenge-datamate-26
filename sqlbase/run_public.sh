#!/usr/bin/env bash
# The 0.63 setup on the public test, without Docker: LoRA v2, few-shot bank with macro examples, one spelling per name, notes before the SQL, saved views and debt layers (sqlbase/config_v6_public), debt routing, unanswerable majorities.
# Usage: bash sqlbase/run_public.sh [OUT]. The file to submit is OUT/submission.jsonl. Set PYTHON if python3 is not the project's venv. About 2.5 h on one 24 GB GPU.
set -eu
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 VLLM_USE_FLASHINFER_SAMPLER=0
PY=${PYTHON:-python3}
OUT=${1:-runs/public}
ENGINE=${ENGINE:-'{"kv_cache_dtype":"fp8","max_num_seqs":128,"max_num_batched_tokens":8192,"attention_backend":"TRITON_ATTN"}'}
mkdir -p "$OUT"
"$PY" sqlbase/predict.py --data data/public_test --output "$OUT" --model sqlbase/models/Qwen3.5-9B --quant fp8 --k 8 --think-k 4 --gpu-mem "${GPU_MEM:-0.92}" --engine "$ENGINE" \
  --views-from sqlbase/config_v6_public --lora "${LORA_DIR:-runs/lora/sql_rft_v2/adapter}" --lora-thinking --fewshot sqlbase/fewshot_bank24m.jsonl \
  --debt-views --debt-route --canon-names --notes --max-tokens 2048 --answer-unanswerable
"$PY" train/check_submission.py "$OUT/submission.jsonl"
