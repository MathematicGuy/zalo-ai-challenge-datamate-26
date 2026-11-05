#!/usr/bin/env bash
# Raw sales tables hidden from the prompt, everything else as the 0.63 setup. Public first, then holdout as a check and as rows for LoRA v3 if hiding is kept. Nothing is submitted here.
set -u
cd "$(dirname "$0")/../.."
export PATH=$PWD/.venv/bin:$PATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 VLLM_USE_FLASHINFER_SAMPLER=0
LOG=runs/analysis/chain_v7.log
ENGINE='{"kv_cache_dtype":"fp8","max_num_seqs":128,"max_num_batched_tokens":8192,"attention_backend":"TRITON_ATTN"}'
COMMON="--model sqlbase/models/Qwen3.5-9B --quant fp8 --k 8 --think-k 4 --gpu-mem 0.92 --lora runs/lora/sql_rft_v2/adapter --lora-thinking --fewshot sqlbase/fewshot_bank24m.jsonl --debt-views --debt-route --save-prompts --canon-names --notes --max-tokens 2048 --hide-raw"
say() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
idle() { local n=0; while pgrep -f "EngineCore" > /dev/null; do sleep 5; n=$((n+1)); if [ $n -gt 60 ]; then kill -9 $(pgrep -f "EngineCore") 2>/dev/null; fi; done; sleep 10; }
until grep -q "chain v6 finished" $LOG; do sleep 20; done
idle
mkdir -p runs/views_v7_public/logs
cp runs/views_v6_public/logs/_*.json runs/views_v7_public/logs/
cp runs/sql_v6_public/logs/_debt_*.json runs/views_v7_public/logs/
OUT=runs/sql_v7_public
say "hide: public with raw sales tables hidden -> $OUT"
mkdir -p $OUT
python3 sqlbase/predict.py --data data/public_test --output $OUT $COMMON --engine "$ENGINE" --views-from runs/views_v7_public > $OUT.log 2>&1
say "hide public exit $?: $(ls $OUT/logs/q_*.json 2>/dev/null | wc -l) logs"
python3 runs/analysis/apply_review.py $OUT none runs/sub_v7 --data data/public_test --answer-unanswerable 2>&1 | grep -v Warning | cut -c1-170 | tee -a $LOG
python3 train/check_submission.py runs/sub_v7/submission.jsonl 2>&1 | grep -v Warning | tail -n 2 | tee -a $LOG
python3 runs/analysis/flags.py public_v7 $OUT runs/views_v7_public --data data/public_test --canon 2>&1 | grep -v Warning | cut -c1-150 > runs/debug/public_v7.txt
say "hide public finished"
idle
OUT=runs/sql_v7_holdout632
say "hide: holdout with raw sales tables hidden -> $OUT"
mkdir -p $OUT
python3 sqlbase/predict.py --data data/train --output $OUT $COMMON --think-below 0.9 --engine "$ENGINE" --views-from runs/views_qty_holdout --ids splits/holdout600.txt > $OUT.log 2>&1
say "hide holdout exit $?: PA N right wrong abstain = $(python3 runs/analysis/score_sub.py $OUT/submission.jsonl splits/holdout600.txt 2>/dev/null)"
python3 runs/analysis/trace_compare.py runs/sql_v6_holdout632 $OUT 2>&1 | grep -v Warning | cut -c1-175 | tee -a $LOG
say "chain hide finished"
