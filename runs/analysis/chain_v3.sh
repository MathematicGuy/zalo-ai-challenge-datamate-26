#!/usr/bin/env bash
# LoRA v3 on all 30 train shops, after chain_canon. 9B + LoRA v2 samples on the 24 other shops (the 6 holdout shops come from chain_canon's run), fresh views with one spelling per name
# and base-unit qty; Qwen3.8-27B on the train questions no 9B sample solved, every chunk, at most 8 hours, skipped if it does not run; rows filtered by gold and by route, QLoRA continued from v2,
# then fresh samples on public. The 27B runs on train only and is not used at inference. Nothing is submitted here.
set -u
cd "$(dirname "$0")/../.."
export PATH=$PWD/.venv/bin:$PATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 VLLM_USE_FLASHINFER_SAMPLER=0
LOG=runs/analysis/chain_v7.log
ENGINE='{"kv_cache_dtype":"fp8","max_num_seqs":128,"max_num_batched_tokens":8192,"attention_backend":"TRITON_ATTN"}'
BANK=sqlbase/fewshot_bank24.jsonl
say() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
idle() { local n=0; while pgrep -f "EngineCore" > /dev/null; do sleep 5; n=$((n+1)); if [ $n -gt 60 ]; then kill -9 $(pgrep -f "EngineCore") 2>/dev/null; fi; done; sleep 10; }
ENGINE27='{"kv_cache_dtype":"fp8","max_num_seqs":16,"max_num_batched_tokens":4096,"attention_backend":"TRITON_ATTN"}'

until grep -q "chain canon finished" $LOG; do sleep 20; done
idle
OUT=runs/sql_v5_gen24
say "v3: fresh views and samples on the 24 other train shops -> $OUT"
mkdir -p $OUT
python3 sqlbase/predict.py --data data/train --output $OUT --model sqlbase/models/Qwen3.5-9B --quant fp8 --k 8 --think-k 4 --think-below 0.9 --gpu-mem 0.92 --engine "$ENGINE" \
  --lora runs/lora/sql_rft_v2/adapter --lora-thinking --fewshot $BANK --debt-views --debt-route --save-prompts --canon-names \
  --views-from runs/views_v3_train --ids runs/analysis/v3_gen_ids.txt > $OUT.log 2>&1
say "v3 generation exit $?: $(ls $OUT/logs/q_*.json 2>/dev/null | wc -l) logs, PA N right wrong abstain = $(python3 runs/analysis/score_sub.py $OUT/submission.jsonl runs/analysis/v3_gen_ids.txt 2>/dev/null)"
grep -aE "^\[views\]" $OUT.log | cut -c1-120 >> $LOG
idle
GEN=$OUT
HOLD=runs/sql_v5_holdout632; [ -s $HOLD/trace.jsonl ] || HOLD=runs/sql_v4_holdout632_canon
say "v3, against older runs (log only): $(python3 runs/analysis/v3_guard.py $GEN runs/sql_v3_gen30 runs/sql_v3_gen30b 2>/dev/null)"
python3 runs/analysis/v3_hard_ids.py $GEN $HOLD 2>&1 | grep -v Warning | tee -a $LOG
T0=$SECONDS; TEACH=""
if [ -f sqlbase/models/Qwen3.8-27B-int4/config.json ]; then
  for f in runs/analysis/v3_teacher_ids_*.txt; do
    [ -f "$f" ] || break
    [ $((SECONDS-T0)) -gt 28800 ] && break
    i=$(basename "$f" .txt); OUT=runs/sql_v5_${i#v3_}
    mkdir -p $OUT
    timeout 7200 python3 sqlbase/predict.py --data data/train --output $OUT --model sqlbase/models/Qwen3.8-27B-int4 --k 2 --think-k 4 --think-below 1.01 --gpu-mem 0.95 --engine "$ENGINE27" \
      --fewshot $BANK --debt-views --debt-route --save-prompts --canon-names --views-from runs/views_v3_teacher --ids "$f" > $OUT.log 2>&1
    rc=$?
    say "27B chunk $i exit $rc after $((SECONDS-T0))s: $(wc -l < $OUT/trace.jsonl 2>/dev/null || echo 0) questions, PA N right wrong abstain = $(python3 runs/analysis/score_sub.py $OUT/submission.jsonl "$f" 2>/dev/null)"
    idle
    [ -s $OUT/trace.jsonl ] && TEACH="$TEACH $OUT"
    [ $rc -ne 0 ] && break
    say "27B chunk $i: a right sample on $(python3 runs/analysis/teacher_yield.py $OUT 2>/dev/null)% of its questions"
  done
else
  say "27B: no weights, skipped"
fi
python3 runs/analysis/build_sft_v3.py $GEN $HOLD --teacher $TEACH --fewshot $BANK --out data/sft/sql_rft_v3.jsonl --max 4000 --easy 800 2>&1 | grep -v Warning | cut -c1-400 | tee -a $LOG
ROWS=$(wc -l < data/sft/sql_rft_v3.jsonl 2>/dev/null || echo 0)
if [ "$ROWS" -lt 800 ]; then say "v3: only $ROWS rows, training NOT started"; exit 1; fi
say "v3: QLoRA continued from v2 on $ROWS rows -> runs/lora/sql_rft_v3"
.venv-train/bin/python train/train_lora.py data/sft/sql_rft_v3.jsonl --out runs/lora/sql_rft_v3 --model sqlbase/models/Qwen3.5-9B --init-adapter runs/lora/sql_rft_v2/adapter \
  --max-len 20000 --r 16 --alpha 32 --lr 5e-5 --epochs 1 --grad-accum 8 --end '<|im_end|>' --no-thinking > runs/lora/sql_rft_v3.log 2>&1
say "v3 training exit $?: $(grep -E 'steps in|examples,|continuing from' runs/lora/sql_rft_v3.log | tr '\n' ' ' | cut -c1-240)"
if [ ! -f runs/lora/sql_rft_v3/adapter/adapter_model.safetensors ]; then say "v3: no adapter, public run NOT started"; exit 1; fi
sleep 30
mkdir -p runs/views_v5_public/logs
cp runs/views_canon_public/logs/_*.json runs/views_v5_public/logs/
cp runs/sql_v4_two/logs/_views_dev_011.json runs/sql_v4_two/logs/_views_dev_014.json runs/sql_v4_two/logs/_debt_dev_014.json runs/views_v5_public/logs/ 2>/dev/null
OUT=runs/sql_v5_public
say "v3: fresh samples on public with LoRA v3 -> $OUT"
mkdir -p $OUT
python3 sqlbase/predict.py --data data/public_test --output $OUT --model sqlbase/models/Qwen3.5-9B --quant fp8 --k 8 --think-k 4 --gpu-mem 0.92 --engine "$ENGINE" \
  --lora runs/lora/sql_rft_v3/adapter --lora-thinking --fewshot $BANK --debt-views --debt-route --save-prompts --canon-names --views-from runs/views_v5_public > $OUT.log 2>&1
say "v3 public exit $?: $(ls $OUT/logs/q_*.json 2>/dev/null | wc -l) logs"
python3 runs/analysis/apply_review.py $OUT none runs/sub_v5 --data data/public_test --answer-unanswerable 2>&1 | grep -v Warning | cut -c1-170 | tee -a $LOG
python3 train/check_submission.py runs/sub_v5/submission.jsonl 2>&1 | grep -v Warning | tail -n 2 | tee -a $LOG
python3 runs/analysis/flags.py public_v5 $OUT runs/views_v5_public --data data/public_test --canon 2>&1 | grep -v Warning | cut -c1-150 > runs/debug/public_v5.txt
say "chain v3 finished"
