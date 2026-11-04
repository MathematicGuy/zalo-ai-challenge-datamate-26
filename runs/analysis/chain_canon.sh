#!/usr/bin/env bash
# After the public run: its saved samples on one-spelling tables and price-factor views, fresh samples for the two shops that had no view, one file.
set -u
cd "$(dirname "$0")/../.."
export PATH=$PWD/.venv/bin:$PATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 VLLM_USE_FLASHINFER_SAMPLER=0
LOG=runs/analysis/chain_v7.log
ENGINE='{"kv_cache_dtype":"fp8","max_num_seqs":128,"max_num_batched_tokens":8192,"attention_backend":"TRITON_ATTN"}'
COMMON="--model sqlbase/models/Qwen3.5-9B --quant fp8 --k 8 --think-k 4 --gpu-mem 0.92 --lora runs/lora/sql_rft_v2/adapter --lora-thinking --fewshot sqlbase/fewshot_bank24.jsonl --debt-views --debt-route --save-prompts --canon-names"
say() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
until grep -q "chain fulleval finished" $LOG; do sleep 20; done
while pgrep -f "EngineCore" > /dev/null; do sleep 5; done
sleep 10
say "canon: saved public samples on one-spelling tables and price-factor views"
python3 runs/analysis/reexec_views.py runs/sql_v4_public runs/views_canon_public runs/sql_v4_public_canon --data data/public_test --canon > runs/sql_v4_public_canon.log 2>&1
say "canon re-exec exit $?: $(tail -n 1 runs/sql_v4_public_canon.log | cut -c1-110)"
OUT=runs/sql_v4_two
say "canon: fresh views and samples on the two shops without a view -> $OUT"
mkdir -p $OUT
python3 sqlbase/predict.py --data data/public_test --output $OUT $COMMON --engine "$ENGINE" --views-from runs/views_canon_fresh --ids runs/analysis/two_shops_ids.txt > $OUT.log 2>&1
say "two shops exit $?: $(ls $OUT/logs/q_*.json 2>/dev/null | wc -l) logs"
grep -aE "^\[views\]|^\[debt" $OUT.log | cut -c1-170 | tee -a $LOG
if [ -s $OUT/trace.jsonl ]; then
  python3 runs/analysis/merge_traces.py runs/sql_v4_public_canon $OUT runs/sql_v4_final 2>&1 | tee -a $LOG
else
  say "two shops gave no trace: the file keeps their old samples"
  mkdir -p runs/sql_v4_final; cp runs/sql_v4_public_canon/trace.jsonl runs/sql_v4_final/trace.jsonl
fi
python3 runs/analysis/apply_review.py runs/sql_v4_public none runs/sub_v4_plain --data data/public_test --answer-unanswerable 2>&1 | grep -v Warning | tee -a $LOG
python3 runs/analysis/apply_review.py runs/sql_v4_final none runs/sub_v4_final --data data/public_test --answer-unanswerable 2>&1 | grep -v Warning | tee -a $LOG
python3 train/check_submission.py runs/sub_v4_final/submission.jsonl 2>&1 | grep -v Warning | tail -n 2 | tee -a $LOG
python3 runs/analysis/flags.py public_v4_final runs/sql_v4_final runs/views_canon_public --data data/public_test --canon 2>&1 | grep -v Warning | cut -c1-150 > runs/debug/public_v4_final.txt
say "canon public finished"
sleep 10
OUT=runs/sql_v5_holdout632
say "canon: fresh samples on holdout, one spelling per name -> $OUT"
mkdir -p $OUT
python3 sqlbase/predict.py --data data/train --output $OUT $COMMON --think-below 0.9 --engine "$ENGINE" --views-from runs/views_qty_holdout --ids splits/holdout600.txt > $OUT.log 2>&1
say "holdout v5 exit $?: PA N right wrong abstain = $(python3 runs/analysis/score_sub.py $OUT/submission.jsonl splits/holdout600.txt 2>/dev/null)"
python3 runs/analysis/trace_compare.py runs/sql_v4_holdout632 $OUT 2>&1 | grep -v Warning | cut -c1-175 | tee -a $LOG
python3 runs/analysis/flags.py holdout_v5 $OUT runs/views_qty_holdout --data data/train --labels --canon 2>&1 | grep -v Warning | cut -c1-185 > runs/debug/holdout_v5.txt
say "chain canon finished"
