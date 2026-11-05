#!/usr/bin/env bash
# LoRA v3 on all 30 train shops, after chain_canon. 9B + LoRA v2 samples on the 24 other shops (the 6 holdout shops come from chain_canon's run), fresh views with one spelling per name
# and base-unit qty; Qwen3.8-27B on the train questions no 9B sample solved, every chunk, at most 8 hours, skipped if it does not run; rows filtered by gold and by route, QLoRA continued from v2,
# then fresh samples on public. A test before generation decides whether plain rows get short notes (runs/analysis/notes_test.py). The 27B runs on train only and is not used at inference. Nothing is submitted here.
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

until grep -q "chain hide finished" $LOG; do sleep 20; done
idle
# the public score of runs/sub_v7 decides: touch runs/analysis/v3_use_hide to generate and train with raw tables hidden, runs/analysis/v3_go to start at once; at most 2 h of waiting
T0=$SECONDS; until [ -f runs/analysis/v3_go ] || [ $((SECONDS-T0)) -gt 7200 ]; do sleep 30; done
BANK=sqlbase/fewshot_bank24m.jsonl
NOTES="--notes --max-tokens 2048"
HOLD6=runs/sql_v6_holdout632
if [ -f runs/analysis/v3_use_hide ]; then NOTES="$NOTES --hide-raw"; HOLD6=runs/sql_v7_holdout632; fi
say "v3: starts with: $NOTES"
OUT=runs/sql_v5_gen24
say "v3: fresh views and samples on the 24 other train shops -> $OUT"
mkdir -p $OUT
python3 sqlbase/predict.py --data data/train --output $OUT --model sqlbase/models/Qwen3.5-9B --quant fp8 --k 8 --think-k 4 --think-below 0.9 --gpu-mem 0.92 --engine "$ENGINE" $NOTES \
  --lora runs/lora/sql_rft_v2/adapter --lora-thinking --fewshot $BANK --debt-views --debt-route --save-prompts --canon-names \
  --views-from runs/views_v3_train --ids runs/analysis/v3_gen_ids.txt > $OUT.log 2>&1
say "v3 generation exit $?: $(ls $OUT/logs/q_*.json 2>/dev/null | wc -l) logs, PA N right wrong abstain = $(python3 runs/analysis/score_sub.py $OUT/submission.jsonl runs/analysis/v3_gen_ids.txt 2>/dev/null)"
grep -aE "^\[views\]" $OUT.log | cut -c1-120 >> $LOG
idle
GEN=$OUT
HOLD=$HOLD6; [ -s $HOLD/trace.jsonl ] || HOLD=runs/sql_v5_holdout632
say "v3, against older runs (log only): $(python3 runs/analysis/v3_guard.py $GEN runs/sql_v3_gen30 runs/sql_v3_gen30b 2>/dev/null)"
python3 runs/analysis/v3_hard_ids.py $GEN $HOLD 2>&1 | grep -v Warning | tee -a $LOG
T0=$SECONDS; TEACH=""
if [ -f sqlbase/models/Qwen3.8-27B-int4/config.json ]; then
  CTX=49152; BUD=24576  # a long budget, lowered only if the engine cannot start with it on 24 GB
  for f in runs/analysis/v3_teacher_ids_*.txt; do
    [ -f "$f" ] || break
    [ $((SECONDS-T0)) -gt 28800 ] && break
    i=$(basename "$f" .txt); OUT=runs/sql_v5_${i#v3_}
    STOP=0
    while :; do
      mkdir -p $OUT
      timeout 14400 python3 sqlbase/predict.py --data data/train --output $OUT --model sqlbase/models/Qwen3.8-27B-int4 --k 2 --think-k 4 --think-below 1.01 --gpu-mem 0.95 --engine "$ENGINE27" $NOTES \
        --context $CTX --think-budget $BUD --fewshot $BANK --debt-views --debt-route --save-prompts --canon-names --views-from runs/views_v3_teacher --ids "$f" > $OUT.log 2>&1
      rc=$?
      say "27B chunk $i, context $CTX, thinking budget $BUD, exit $rc after $((SECONDS-T0))s: $(wc -l < $OUT/trace.jsonl 2>/dev/null || echo 0) questions"
      idle
      [ -s $OUT/trace.jsonl ] && break
      [ $rc -eq 124 ] && break
      case $CTX in
        49152) CTX=32768; BUD=16384 ;;
        32768) CTX=24576; BUD=8192 ;;
        *) STOP=1; break ;;
      esac
    done
    [ $STOP -eq 1 ] && { say "27B: the engine does not start, skipped"; break; }
    if [ -s $OUT/trace.jsonl ]; then
      TEACH="$TEACH $OUT"
      say "27B chunk $i: a right sample on $(python3 runs/analysis/teacher_yield.py $OUT 2>/dev/null)% of its questions"
    fi
  done
else
  say "27B: no weights, skipped"
fi
python3 runs/analysis/build_sft_v3.py $GEN $HOLD --teacher $TEACH --fewshot $BANK --out data/sft/sql_rft_v3.jsonl --max 4000 --easy 800 2>&1 | grep -v Warning | cut -c1-400 | tee -a $LOG
ROWS=$(wc -l < data/sft/sql_rft_v3.jsonl 2>/dev/null || echo 0)
SFT=data/sft/sql_rft_v3.jsonl
if [ -f runs/analysis/v3_use_notes ]; then
  say "v3: adding notes to the plain rows"
  python3 runs/analysis/add_notes.py $SFT data/sft/sql_rft_v3_notes.jsonl 2>&1 | grep -aE "^notes on|Traceback|Error" | cut -c1-160 | tee -a $LOG
  idle
  [ -s data/sft/sql_rft_v3_notes.jsonl ] && SFT=data/sft/sql_rft_v3_notes.jsonl
fi
if [ "$ROWS" -lt 800 ]; then say "v3: only $ROWS rows, training NOT started"; exit 1; fi
say "v3: QLoRA continued from v2 on $ROWS rows -> runs/lora/sql_rft_v3"
.venv-train/bin/python train/train_lora.py $SFT --out runs/lora/sql_rft_v3 --model sqlbase/models/Qwen3.5-9B --init-adapter runs/lora/sql_rft_v2/adapter \
  --max-len 32768 --r 16 --alpha 32 --lr 5e-5 --epochs 1 --grad-accum 8 --end '<|im_end|>' --no-thinking > runs/lora/sql_rft_v3.log 2>&1
say "v3 training exit $?: $(grep -E 'steps in|examples,|continuing from' runs/lora/sql_rft_v3.log | tr '\n' ' ' | cut -c1-240)"
if [ ! -f runs/lora/sql_rft_v3/adapter/adapter_model.safetensors ]; then
  say "v3: no adapter at 32768 tokens per row, training again at 20000"
  sleep 30
  .venv-train/bin/python train/train_lora.py $SFT --out runs/lora/sql_rft_v3 --model sqlbase/models/Qwen3.5-9B --init-adapter runs/lora/sql_rft_v2/adapter \
    --max-len 20000 --r 16 --alpha 32 --lr 5e-5 --epochs 1 --grad-accum 8 --end '<|im_end|>' --no-thinking > runs/lora/sql_rft_v3_retry.log 2>&1
  say "v3 second training exit $?: $(grep -E 'steps in|examples,' runs/lora/sql_rft_v3_retry.log | tr '\n' ' ' | cut -c1-200)"
fi
if [ ! -f runs/lora/sql_rft_v3/adapter/adapter_model.safetensors ]; then say "v3: no adapter, public run NOT started"; exit 1; fi
sleep 30
OUT=runs/sql_v5_public
say "v3: fresh samples on public with LoRA v3 -> $OUT"
mkdir -p $OUT
python3 sqlbase/predict.py --data data/public_test --output $OUT --model sqlbase/models/Qwen3.5-9B --quant fp8 --k 8 --think-k 4 --gpu-mem 0.92 --engine "$ENGINE" \
  --context 49152 --think-budget 16384 $NOTES --lora runs/lora/sql_rft_v3/adapter --lora-thinking --fewshot $BANK --debt-views --debt-route --save-prompts --canon-names --views-from runs/views_v7_public > $OUT.log 2>&1
say "v3 public exit $?: $(ls $OUT/logs/q_*.json 2>/dev/null | wc -l) logs"
python3 runs/analysis/apply_review.py $OUT none runs/sub_v5 --data data/public_test --answer-unanswerable 2>&1 | grep -v Warning | cut -c1-170 | tee -a $LOG
python3 train/check_submission.py runs/sub_v5/submission.jsonl 2>&1 | grep -v Warning | tail -n 2 | tee -a $LOG
python3 runs/analysis/flags.py public_v5 $OUT runs/views_v7_public --data data/public_test --canon 2>&1 | grep -v Warning | cut -c1-150 > runs/debug/public_v5.txt
say "chain v3 finished"
