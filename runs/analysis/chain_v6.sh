#!/usr/bin/env bash
# No training: plain samples start with short notes, the few-shot bank has examples that call the debt macros. Public first, for a file by morning; then holdout, as a check over easy questions too and as LoRA v3 rows.
set -u
cd "$(dirname "$0")/../.."
export PATH=$PWD/.venv/bin:$PATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 VLLM_USE_FLASHINFER_SAMPLER=0
LOG=runs/analysis/chain_v7.log
ENGINE='{"kv_cache_dtype":"fp8","max_num_seqs":128,"max_num_batched_tokens":8192,"attention_backend":"TRITON_ATTN"}'
COMMON="--model sqlbase/models/Qwen3.5-9B --quant fp8 --k 8 --think-k 4 --gpu-mem 0.92 --lora runs/lora/sql_rft_v2/adapter --lora-thinking --fewshot sqlbase/fewshot_bank24m.jsonl --debt-views --debt-route --save-prompts --canon-names --notes --max-tokens 2048"
say() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
idle() { local n=0; while pgrep -f "EngineCore" > /dev/null; do sleep 5; n=$((n+1)); if [ $n -gt 60 ]; then kill -9 $(pgrep -f "EngineCore") 2>/dev/null; fi; done; sleep 10; }
idle
mkdir -p runs/views_v6_public/logs
cp runs/views_canon_public/logs/_*.json runs/views_v6_public/logs/
cp runs/sql_v4_two/logs/_views_dev_011.json runs/sql_v4_two/logs/_views_dev_014.json runs/views_v6_public/logs/
OUT=runs/sql_v6_public
say "v6: public with notes and the macro bank -> $OUT"
mkdir -p $OUT
python3 sqlbase/predict.py --data data/public_test --output $OUT $COMMON --engine "$ENGINE" --views-from runs/views_v6_public > $OUT.log 2>&1
say "v6 public exit $?: $(ls $OUT/logs/q_*.json 2>/dev/null | wc -l) logs"
grep -aE "^\[debts\]" $OUT.log | cut -c1-120 | tee -a $LOG
python3 runs/analysis/apply_review.py $OUT none runs/sub_v6 --data data/public_test --answer-unanswerable 2>&1 | grep -v Warning | cut -c1-170 | tee -a $LOG
python3 train/check_submission.py runs/sub_v6/submission.jsonl 2>&1 | grep -v Warning | tail -n 2 | tee -a $LOG
python3 runs/analysis/flags.py public_v6 $OUT runs/views_v6_public --data data/public_test --canon 2>&1 | grep -v Warning | cut -c1-150 > runs/debug/public_v6.txt
say "v6 public finished"
idle
OUT=runs/sql_v6_holdout632
say "v6: holdout with notes and the macro bank -> $OUT"
mkdir -p $OUT
python3 sqlbase/predict.py --data data/train --output $OUT $COMMON --think-below 0.9 --engine "$ENGINE" --views-from runs/views_qty_holdout --ids splits/holdout600.txt > $OUT.log 2>&1
say "v6 holdout exit $?: PA N right wrong abstain = $(python3 runs/analysis/score_sub.py $OUT/submission.jsonl splits/holdout600.txt 2>/dev/null)"
python3 runs/analysis/trace_compare.py runs/sql_v5_holdout632 $OUT 2>&1 | grep -v Warning | cut -c1-175 | tee -a $LOG
python3 runs/analysis/flags.py holdout_v6 $OUT runs/views_qty_holdout --data data/train --labels --canon 2>&1 | grep -v Warning | cut -c1-185 > runs/debug/holdout_v6.txt
say "chain v6 finished"
