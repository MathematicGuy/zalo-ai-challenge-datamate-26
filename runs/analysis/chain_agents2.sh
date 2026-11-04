#!/usr/bin/env bash
# Rest of chain_agents.sh once the reader run ends (pid $1), with the two steps that failed on a missing main guard put first, cheapest first.
set -u
cd "$(dirname "$0")/../.."
export PATH=$PWD/.venv/bin:$PATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 VLLM_USE_FLASHINFER_SAMPLER=0
LOG=runs/analysis/chain_v7.log
ENGINE='{"kv_cache_dtype":"fp8","max_num_seqs":128,"max_num_batched_tokens":8192,"attention_backend":"TRITON_ATTN"}'
say() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
while kill -0 "$1" 2>/dev/null; do sleep 10; done
sleep 15
OUT=runs/sql_v2_reader350
say "reader run ended: $(ls $OUT/logs/q_*.json 2>/dev/null | wc -l) logs"
grep -aE "^\[reader\]" $OUT.log | cut -c1-90 | tee -a $LOG
say "agents: flash-sqlite judge over the LoRA v2 holdout trace"
python3 runs/analysis/judge_trace.py runs/sql_v2_holdout632 runs/judge_flash_holdout632.jsonl --data data/train --views-from runs/sql_v2_holdout632 --model sqlbase/models/flash-sqlite > runs/judge_flash_holdout632.log 2>&1
say "judge exit $?: $(tail -n 1 runs/judge_flash_holdout632.log | cut -c1-120)"
python3 runs/analysis/judge_eval.py runs/judge_flash_holdout632.jsonl 2>&1 | grep -v Warning | tee -a $LOG
say "judge eval finished"
sleep 10
say "agents: reviewer over the LoRA v2 holdout trace"
python3 runs/analysis/review_trace.py runs/sql_v2_holdout632 runs/review_v2_holdout632.jsonl --data data/train > runs/review_v2_holdout632.log 2>&1
say "reviewer exit $?: $(tail -n 1 runs/review_v2_holdout632.log | cut -c1-140)"
sleep 10
say "agents: reviewer over the reader run"
python3 runs/analysis/review_trace.py $OUT runs/review_v2_reader350.jsonl --data data/train --views-from runs/sql_v2_holdout632 > runs/review_v2_reader350.log 2>&1
say "reviewer on reader exit $?: $(tail -n 1 runs/review_v2_reader350.log | cut -c1-140)"
python3 runs/analysis/agents_eval.py 2>&1 | grep -v Warning | tee -a $LOG
say "roles eval finished"
sleep 10
OUT=runs/sql_flash_gen350
say "agents: flash-sqlite as a writer, 8 samples on the same questions -> $OUT"
mkdir -p $OUT
python3 sqlbase/predict.py --data data/train --output $OUT --model sqlbase/models/flash-sqlite --quant fp8 --k 8 --think-k 0 \
  --gpu-mem 0.92 --engine "$ENGINE" --views-from runs/sql_v2_holdout632 --fewshot sqlbase/fewshot_bank24.jsonl --debt-views --ids splits/reader_test.txt > $OUT.log 2>&1
say "flash writer exit $?: PA N right wrong abstain = $(python3 runs/analysis/score_sub.py $OUT/submission.jsonl splits/reader_test.txt 2>/dev/null)"
say "chain agents finished"
