#!/usr/bin/env bash
# The exploration agent on the hard holdout questions, after the role tests, then the GPU goes to gen30b.
set -u
cd "$(dirname "$0")/../.."
export PATH=$PWD/.venv/bin:$PATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 VLLM_USE_FLASHINFER_SAMPLER=0
LOG=runs/analysis/chain_v7.log
say() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
until grep -q "chain agents finished" $LOG; do sleep 20; done
sleep 15
say "explore: agent on the hard holdout questions, base weights -> runs/explore_v2_holdout.jsonl"
python3 runs/analysis/explore_trace.py runs/sql_v2_holdout632 runs/explore_v2_holdout.jsonl --data data/train --views-from runs/sql_v2_holdout632 > runs/explore_v2_holdout.log 2>&1
say "explore exit $?: $(tail -n 1 runs/explore_v2_holdout.log | cut -c1-150)"
python3 runs/analysis/explore_eval.py runs/explore_v2_holdout.jsonl 2>&1 | grep -v Warning | tee -a $LOG
say "chain explore finished"
