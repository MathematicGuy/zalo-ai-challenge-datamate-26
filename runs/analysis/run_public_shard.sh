#!/usr/bin/env bash
# sqlbase/run_public.sh flags, one business shard per call: bash runs/analysis/run_public_shard.sh OUT GPU I N [extra flags]
set -eu
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 VLLM_USE_FLASHINFER_SAMPLER=0
PY=${PYTHON:-python3}
OUT=$1 GPU=$2 I=$3 N=$4; shift 4
ENGINE=${ENGINE:-'{"kv_cache_dtype":"fp8","max_num_seqs":128,"max_num_batched_tokens":8192,"attention_backend":"TRITON_ATTN"}'}
mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=$GPU "$PY" "${PREDICT:-sqlbase/predict.py}" --data data/public_test --output "$OUT/shard$I" --shard "$I/$N" --model sqlbase/models/Qwen3.5-9B --quant fp8 --k 8 --think-k 4 \
  --gpu-mem "${GPU_MEM:-0.92}" --engine "$ENGINE" --views-from sqlbase/config_v6_public --lora "${LORA_DIR:-runs/lora/sql_rft_v2/adapter}" --lora-thinking \
  --fewshot sqlbase/fewshot_bank24m.jsonl --debt-views --debt-route --canon-names --notes --max-tokens 2048 --answer-unanswerable "$@" > "$OUT/shard$I.txt" 2>&1
