#!/usr/bin/env bash
# Rest of chain_fulleval.sh once the holdout run ends (pid $1): compare and flags, another try at the two public views the repair left, then fresh samples on public.
set -u
cd "$(dirname "$0")/../.."
export PATH=$PWD/.venv/bin:$PATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 VLLM_USE_FLASHINFER_SAMPLER=0
LOG=runs/analysis/chain_v7.log
ENGINE='{"kv_cache_dtype":"fp8","max_num_seqs":128,"max_num_batched_tokens":8192,"attention_backend":"TRITON_ATTN"}'
say() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
while kill -0 "$1" 2>/dev/null; do sleep 15; done
sleep 15
OUT=runs/sql_v4_holdout632
say "fulleval holdout ended: PA N right wrong abstain = $(python3 runs/analysis/score_sub.py $OUT/submission.jsonl splits/holdout600.txt 2>/dev/null)"
python3 runs/analysis/trace_compare.py runs/sql_v2_holdout632 runs/sql_v2_qtyauto632 $OUT 2>&1 | grep -v Warning | cut -c1-175 | tee -a $LOG
python3 runs/analysis/flags.py holdout_v4 $OUT runs/views_qty_holdout --data data/train --labels 2>&1 | grep -v Warning | cut -c1-185 > runs/debug/holdout_v4.txt
sed -n '/first stage that broke/,/stage x intent/p' runs/debug/holdout_v4.txt | tee -a $LOG
say "fulleval holdout finished"
sleep 10
say "qty: another try on dev_005 and dev_013, longer budget and 3 repair turns"
python3 runs/analysis/fix_qty.py --data data/public_test --views-from runs/views_qty_public --out runs/views_qty_public --only dev_005,dev_013 --budget 24576 --turns 3 > runs/views_qty_public2.log 2>&1
say "qty retry exit $?"
grep -aE "^\[qty\]|^qty turn" runs/views_qty_public2.log | cut -c1-170 | tee -a $LOG
sleep 10
OUT=runs/sql_v4_public
say "fulleval: fresh samples on public with base-unit views -> $OUT"
mkdir -p $OUT
python3 sqlbase/predict.py --data data/public_test --output $OUT --model sqlbase/models/Qwen3.5-9B --quant fp8 --k 8 --think-k 4 --gpu-mem 0.92 --engine "$ENGINE" \
  --views-from runs/views_qty_public --lora runs/lora/sql_rft_v2/adapter --lora-thinking --fewshot sqlbase/fewshot_bank24.jsonl --debt-views --debt-route --save-prompts > $OUT.log 2>&1
say "fulleval public exit $?: $(ls $OUT/logs/q_*.json 2>/dev/null | wc -l) logs"
python3 train/check_submission.py $OUT/submission.jsonl 2>&1 | grep -v Warning | tail -n 2 | tee -a $LOG
python3 runs/analysis/flags.py public_v4 $OUT runs/views_qty_public --data data/public_test 2>&1 | grep -v Warning | cut -c1-150 > runs/debug/public_v4.txt
say "chain fulleval finished"
