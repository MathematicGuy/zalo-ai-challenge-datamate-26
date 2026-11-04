#!/usr/bin/env bash
# After the 4B test, cheapest first: the flash-sqlite judge over the LoRA v2 holdout trace; the reviewer over the same trace; the reader with fresh samples on
# 350 questions (all far and mid, 40 near) and the reviewer on those; flash-sqlite as a second writer on the same 350. A failed step does not hold the GPU.
set -u
cd "$(dirname "$0")/../.."
export PATH=$PWD/.venv/bin:$PATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 VLLM_USE_FLASHINFER_SAMPLER=0
LOG=runs/analysis/chain_v7.log
ENGINE='{"kv_cache_dtype":"fp8","max_num_seqs":128,"max_num_batched_tokens":8192,"attention_backend":"TRITON_ATTN"}'
say() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
until grep -q "chain q4b finished" $LOG && grep -q "flash download exit" $LOG; do sleep 20; done
sleep 20
say "agents: flash-sqlite judge over the LoRA v2 holdout trace"
python3 runs/analysis/judge_trace.py runs/sql_v2_holdout632 runs/judge_flash_holdout632.jsonl --data data/train --views-from runs/sql_v2_holdout632 --model sqlbase/models/flash-sqlite > runs/judge_flash_holdout632.log 2>&1
say "judge exit $?: $(tail -n 1 runs/judge_flash_holdout632.log | cut -c1-120)"
python3 runs/analysis/judge_eval.py runs/judge_flash_holdout632.jsonl 2>&1 | grep -v Warning | tee -a $LOG
sleep 10
say "agents: reviewer over the LoRA v2 holdout trace"
python3 runs/analysis/review_trace.py runs/sql_v2_holdout632 runs/review_v2_holdout632.jsonl --data data/train > runs/review_v2_holdout632.log 2>&1
say "reviewer exit $?: $(tail -n 1 runs/review_v2_holdout632.log | cut -c1-140)"
sleep 10
OUT=runs/sql_v2_reader350
say "agents: reader + LoRA v2, 8 plain + 4 thinking on $(wc -l < splits/reader_test.txt) questions -> $OUT"
mkdir -p $OUT
python3 sqlbase/predict.py --data data/train --output $OUT --model sqlbase/models/Qwen3.5-9B --quant fp8 --k 8 --think-k 4 \
  --gpu-mem 0.92 --engine "$ENGINE" --views-from runs/sql_v2_holdout632 --lora runs/lora/sql_rft_v2/adapter --lora-thinking --fewshot sqlbase/fewshot_bank24.jsonl --debt-views --reader --ids splits/reader_test.txt > $OUT.log 2>&1
say "reader exit $?: $(ls $OUT/logs/q_*.json 2>/dev/null | wc -l) logs"
grep -aE "^\[reader\]" $OUT.log | cut -c1-90 | tee -a $LOG
sleep 10
say "agents: reviewer over the reader run"
python3 runs/analysis/review_trace.py $OUT runs/review_v2_reader350.jsonl --data data/train --views-from runs/sql_v2_holdout632 > runs/review_v2_reader350.log 2>&1
say "reviewer on reader exit $?: $(tail -n 1 runs/review_v2_reader350.log | cut -c1-140)"
python3 runs/analysis/agents_eval.py 2>&1 | grep -v Warning | tee -a $LOG
sleep 10
OUT=runs/sql_flash_gen350
say "agents: flash-sqlite as a writer, 8 samples on the same questions -> $OUT"
mkdir -p $OUT
python3 sqlbase/predict.py --data data/train --output $OUT --model sqlbase/models/flash-sqlite --quant fp8 --k 8 --think-k 0 \
  --gpu-mem 0.92 --engine "$ENGINE" --views-from runs/sql_v2_holdout632 --fewshot sqlbase/fewshot_bank24.jsonl --debt-views --ids splits/reader_test.txt > $OUT.log 2>&1
say "flash writer exit $?: PA N right wrong abstain = $(python3 runs/analysis/score_sub.py $OUT/submission.jsonl splits/reader_test.txt 2>/dev/null)"
say "chain agents finished"
