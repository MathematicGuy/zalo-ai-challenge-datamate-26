#!/usr/bin/env bash
# The reviewer on the public questions the 0.491 vote leaves unanswered, from the saved samples of that run; gen30b gives up the GPU at a shop boundary.
set -u
cd "$(dirname "$0")/../.."
export PATH=$PWD/.venv/bin:$PATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 VLLM_USE_FLASHINFER_SAMPLER=0
LOG=runs/analysis/chain_v7.log
say() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
GEN_PY=$1 GEN_ENGINE=$2 DONE=$3
while [ "$(grep -acE '^\[biz_' runs/sql_v3_gen30b.log)" -le "$DONE" ] && kill -0 $GEN_PY 2>/dev/null; do sleep 10; done
[ "$(ps -p $GEN_PY -o cmd= | grep -c 'sql_v3_gen30b')" = 1 ] && kill $GEN_PY
sleep 20
[ "$(ps -p $GEN_ENGINE -o cmd= | grep -c 'EngineCore')" = 1 ] && kill -9 $GEN_ENGINE
for _ in $(seq 1 12); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1500 ] && break; sleep 5; done
say "pubreview: gen30b stopped at $(wc -l < runs/sql_v3_gen30b/trace.jsonl) questions, GPU $(nvidia-smi --query-gpu=memory.used --format=csv,noheader) used"
python3 runs/analysis/review_trace.py runs/sql_v2_public runs/review_v2_public.jsonl --data data/public_test --views-from sqlbase/config_v2_public --only-abstained > runs/review_v2_public.log 2>&1
say "pubreview exit $?: $(grep -a 'reviewed .* in' runs/review_v2_public.log | tail -n 1 | cut -c1-120)"
python3 runs/analysis/apply_review.py runs/sql_v2_public runs/review_v2_public.jsonl runs/sql_v2_public_reviewed --data data/public_test 2>&1 | grep -v Warning | tee -a $LOG
python3 train/check_submission.py runs/sql_v2_public_reviewed/submission.jsonl 2>&1 | grep -v Warning | tail -n 2 | tee -a $LOG
say "chain pubreview finished"
