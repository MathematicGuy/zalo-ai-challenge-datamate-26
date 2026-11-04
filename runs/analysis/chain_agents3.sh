#!/usr/bin/env bash
# Rest of the role tests once the reviewer run ends (pid $1): the judge again with its probabilities read correctly, then the reviewer on the reader run, then flash-sqlite as a writer.
set -u
cd "$(dirname "$0")/../.."
export PATH=$PWD/.venv/bin:$PATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 VLLM_USE_FLASHINFER_SAMPLER=0
LOG=runs/analysis/chain_v7.log
ENGINE='{"kv_cache_dtype":"fp8","max_num_seqs":128,"max_num_batched_tokens":8192,"attention_backend":"TRITON_ATTN"}'
say() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
while kill -0 "$1" 2>/dev/null; do sleep 10; done
sleep 15
say "reviewer run ended: $(tail -n 1 runs/review_v2_holdout632.log | cut -c1-140)"
say "agents: flash-sqlite judge again, every P(Yes) was 0 before (Yes missing from the top logprobs)"
python3 runs/analysis/judge_trace.py runs/sql_v2_holdout632 runs/judge_flash_holdout632.jsonl --data data/train --views-from runs/sql_v2_holdout632 --model sqlbase/models/flash-sqlite > runs/judge_flash_holdout632.log 2>&1
say "judge exit $?: $(grep -a 'judged .* groups' runs/judge_flash_holdout632.log | tail -n 1 | cut -c1-100)"
python3 runs/analysis/judge_eval.py runs/judge_flash_holdout632.jsonl 2>&1 | grep -v Warning | tee -a $LOG
say "judge eval finished"
sleep 10
OUT=runs/sql_v2_reader350
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
