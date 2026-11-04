#!/usr/bin/env bash
# The reviewer finished 35 of 125 replies inside 4096 thinking tokens and picked right in 32 of them: the other 90 again with 24576, after the agent test.
set -u
cd "$(dirname "$0")/../.."
export PATH=$PWD/.venv/bin:$PATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 VLLM_USE_FLASHINFER_SAMPLER=0
LOG=runs/analysis/chain_v7.log
say() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
until grep -q "chain explore finished" $LOG; do sleep 20; done
sleep 15
say "reviewer again on the replies cut at 4096 tokens, budget 24576"
python3 runs/analysis/review_trace.py runs/sql_v2_holdout632 runs/review_v2_holdout632.jsonl --data data/train --retry-from runs/review_v2_holdout632_b4096.jsonl --think-budget 24576 --context 49152 > runs/review_long.log 2>&1
say "reviewer long exit $?: $(grep -a 'reviewed .* in' runs/review_long.log | tail -n 1 | cut -c1-120)"
python3 runs/analysis/agents_eval.py 2>&1 | grep -v Warning | sed -n '/reviewer on LoRA v2/,/^reader:/p' | tee -a $LOG
say "chain longreview finished"
