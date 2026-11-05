#!/usr/bin/env bash
# The public setup (run_public.sh flags) on the 632-question business holdout of train, one shard per GPU, then scored.
# Usage: bash runs/analysis/run_holdout.sh OUT [extra predict.py flags]. Views/debt layers are drafted fresh unless --views-from is given.
set -eu
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 VLLM_USE_FLASHINFER_SAMPLER=0
PY=${PYTHON:-python3}
OUT=$1; shift
NG=${NUM_GPUS:-$(nvidia-smi -L | wc -l)}
ENGINE=${ENGINE:-'{"kv_cache_dtype":"fp8","max_num_seqs":128,"max_num_batched_tokens":8192,"attention_backend":"TRITON_ATTN"}'}
mkdir -p "$OUT"
for i in $(seq 0 $((NG - 1))); do
  CUDA_VISIBLE_DEVICES=$i "$PY" "${PREDICT:-sqlbase/predict.py}" --data data/train --ids splits/holdout600.txt --output "$OUT/shard$i" --shard "$i/$NG" \
    --model sqlbase/models/Qwen3.5-9B --quant fp8 --k 8 --think-k 4 --gpu-mem "${GPU_MEM:-0.92}" --engine "$ENGINE" \
    --lora "${LORA_DIR:-runs/lora/sql_rft_v2/adapter}" --lora-thinking --fewshot sqlbase/fewshot_bank24.jsonl \
    --debt-views --debt-route --canon-names --answer-unanswerable "$@" > "$OUT/shard$i.txt" 2>&1 &
done
wait
