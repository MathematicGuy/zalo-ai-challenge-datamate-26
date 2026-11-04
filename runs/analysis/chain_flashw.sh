#!/usr/bin/env bash
# flash-sqlite as a second writer on the 350 test questions, after the long reviewer test; then the GPU goes to gen30b.
set -u
cd "$(dirname "$0")/../.."
export PATH=$PWD/.venv/bin:$PATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 VLLM_USE_FLASHINFER_SAMPLER=0
LOG=runs/analysis/chain_v7.log
say() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
until grep -q "chain longreview finished" $LOG; do sleep 20; done
sleep 15
OUT=runs/sql_flash_gen350
say "flash-sqlite as a writer, 8 samples on 350 questions -> $OUT"
mkdir -p $OUT
python3 sqlbase/predict.py --data data/train --output $OUT --model sqlbase/models/flash-sqlite --quant fp8 --k 8 --think-k 0 \
  --gpu-mem 0.92 --engine '{"kv_cache_dtype":"fp8","max_num_seqs":128,"max_num_batched_tokens":8192,"attention_backend":"TRITON_ATTN"}' \
  --views-from runs/sql_v2_holdout632 --fewshot sqlbase/fewshot_bank24.jsonl --debt-views --ids splits/reader_test.txt > $OUT.log 2>&1
say "flash writer exit $?: PA N right wrong abstain = $(python3 runs/analysis/score_sub.py $OUT/submission.jsonl splits/reader_test.txt 2>/dev/null)"
say "chain flashw finished"
