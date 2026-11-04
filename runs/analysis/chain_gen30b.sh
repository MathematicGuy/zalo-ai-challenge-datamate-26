#!/usr/bin/env bash
# Resumes gen30 after a GPU pause: the same settings on the train questions not yet traced in runs/sql_v3_gen30 -> runs/sql_v3_gen30b. Waits for $1 in the chain log.
set -u
cd "$(dirname "$0")/../.."
export PATH=$PWD/.venv/bin:$PATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 VLLM_USE_FLASHINFER_SAMPLER=0
LOG=runs/analysis/chain_v7.log
say() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
until grep -q "$1" $LOG; do sleep 30; done
sleep 300
python3 - <<'PY'
import json
done = set()
for l in open("runs/sql_v3_gen30/trace.jsonl", encoding="utf-8"):
    try: done.add(json.loads(l)["id"])
    except Exception: pass
qs = [json.loads(l)["id"] for l in open("data/train/questions.jsonl", encoding="utf-8-sig") if l.strip()]
open("splits/gen30_rest.txt", "w").write("\n".join(i for i in qs if i not in done) + "\n")
print(len(done), "done")
PY
OUT=runs/sql_v3_gen30b
say "gen30b: resume on $(wc -l < splits/gen30_rest.txt) questions -> $OUT"
mkdir -p $OUT
python3 sqlbase/predict.py --data data/train --output $OUT --model sqlbase/models/Qwen3.5-9B --quant fp8 --k 2 --think-k 4 --think-below 1.01 \
  --gpu-mem 0.92 --engine '{"kv_cache_dtype":"fp8","max_num_seqs":128,"max_num_batched_tokens":8192,"attention_backend":"TRITON_ATTN"}' \
  --views-from runs/views_train30 --lora runs/lora/sql_rft_v2/adapter --lora-thinking --fewshot sqlbase/fewshot_bank24.jsonl --debt-views --prompt v3 --save-prompts --ids splits/gen30_rest.txt > $OUT.log 2>&1
say "gen30b exit $?: $(ls $OUT/logs/q_*.json 2>/dev/null | wc -l) logs"
say "chain gen30b finished"
