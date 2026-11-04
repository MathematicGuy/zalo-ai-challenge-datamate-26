#!/usr/bin/env bash
# Qwen3.5-4B as a second model (4.66B, total 14.36B with 9B + LoRA): samples on the 632 holdout questions with the views, debt layers and few-shot
# of the 9B run, to pool offline against its trace (runs/analysis/q4b_pool.py). Takes the GPU from gen30 at a shop boundary; gen30 is resumed separately.
set -u
cd "$(dirname "$0")/../.."
export PATH=$PWD/.venv/bin:$PATH
LOG=runs/analysis/chain_v7.log
say() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
GEN_PY=$1 GEN_ENGINE=$2 DONE=$3
M=sqlbase/models/Qwen3.5-4B
say "q4b: downloading Qwen3.5-4B"
hf download Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --local-dir $M > runs/q4b_download.log 2>&1
say "q4b: download exit $?, $(du -sh $M | cut -f1)"
if ! ls $M/*.safetensors >/dev/null 2>&1; then say "q4b: no weights, stopping; gen30 left running"; exit 1; fi
say "q4b: waiting for gen30 to finish a shop (done so far: $DONE)"
while [ "$(grep -acE '^\[biz_' runs/sql_v3_gen30.log)" -le "$DONE" ] && kill -0 $GEN_PY 2>/dev/null; do sleep 10; done
if [ "$(ps -p $GEN_PY -o cmd= | grep -c 'sql_v3_gen30')" = 1 ]; then kill $GEN_PY; fi
for _ in $(seq 1 12); do
  [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1500 ] && break
  sleep 10
done
if [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -ge 1500 ] && [ "$(ps -p $GEN_ENGINE -o cmd= | grep -c 'EngineCore')" = 1 ]; then kill -9 $GEN_ENGINE; sleep 10; fi
say "q4b: gen30 stopped at $(wc -l < runs/sql_v3_gen30/trace.jsonl) traced questions, GPU $(nvidia-smi --query-gpu=memory.used --format=csv,noheader) used"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 VLLM_USE_FLASHINFER_SAMPLER=0
OUT=runs/sql_q4b_holdout632
say "q4b: Qwen3.5-4B, 8 plain + 4 thinking on hard, 632 holdout questions -> $OUT"
mkdir -p $OUT
python3 sqlbase/predict.py --data data/train --output $OUT --model $M --quant fp8 --k 8 --think-k 4 \
  --gpu-mem 0.92 --engine '{"kv_cache_dtype":"fp8","max_num_seqs":128,"max_num_batched_tokens":8192,"attention_backend":"TRITON_ATTN"}' \
  --views-from runs/sql_v2_holdout632 --fewshot sqlbase/fewshot_bank24.jsonl --debt-views --ids splits/holdout600.txt > $OUT.log 2>&1
say "q4b holdout exit $?: PA N right wrong abstain = $(python3 runs/analysis/score_sub.py $OUT/submission.jsonl splits/holdout600.txt 2>/dev/null)"
say "chain q4b finished"
