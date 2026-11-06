#!/usr/bin/env bash
# Container entry point: one datamate.predict process per GPU (data parallel over shops), every view and debt layer drafted
# fresh, then one /result/submission.jsonl (+ /result/logs/). Each process stops at TIMEOUT_SECONDS; answers written so far count.
set -uo pipefail
cd "$(dirname "$0")"
source ./flags.sh
DATA="${DATA_DIR:-/data}" OUT="${RESULT_DIR:-/result}"
NG="${NUM_GPUS:-$(nvidia-smi -L | wc -l)}"
T="${TIMEOUT_SECONDS:-21000}"
mkdir -p "$OUT/logs"
for i in $(seq 0 $((NG - 1))); do
  CUDA_VISIBLE_DEVICES=$i timeout --signal=TERM --kill-after=15s "${T}s" \
    python3 -m datamate.predict --data "$DATA" --output "$OUT/shard$i" --shard "$i/$NG" --deadline "$(( T - 900 ))" \
    --model "${MODEL_DIR:-/models/Qwen3.5-9B}" --lora "${LORA_DIR:-/models/lora}" --fewshot assets/fewshot_bank.jsonl \
    "${DATAMATE_FLAGS[@]}" --engine "$DATAMATE_ENGINE" "$@" > "$OUT/logs/_shard$i.txt" 2>&1 &
done
wait
python3 -m datamate.submission merge "$OUT" --questions "$DATA/questions.jsonl"
