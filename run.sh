#!/usr/bin/env bash
# Run notebooks/02_agent.ipynb on a split, one shard per GPU, headless (papermill).
#
#   ./run.sh SPLIT OUT [papermill args...]
#   ./run.sh train runs/v3_val300 -p IDS ../splits/val300.txt     # validation
#   ./run.sh public_test runs/v3_public                            # public test -> runs/v3_public/submission.jsonl
#
# SPLIT is "train", "public_test" or a split folder (Docker: /data). Extra args are passed to papermill,
# e.g. -p PROMPT v2 -p K 5 -p TAU 0.6. Each shard's executed notebook and log land in OUT/.
# Env: DATAMATE_NGPU (default: all visible GPUs), DATAMATE_KERNEL (default: python3), and the engine
# knobs read by 02_agent §6 (DATAMATE_GPU_UTIL, DATAMATE_MAX_SEQS, DATAMATE_MAX_LEN, DATAMATE_MODEL ...).
set -euo pipefail
cd "$(dirname "$0")"

SPLIT=${1:?usage: ./run.sh SPLIT OUT [papermill args...]}
OUT=${2:?usage: ./run.sh SPLIT OUT [papermill args...]}
shift 2
case "$OUT" in /*) ;; *) OUT="$PWD/$OUT" ;; esac
case "$SPLIT" in /*|train|public_test) ;; *) SPLIT="$PWD/$SPLIT" ;; esac

NGPU=${DATAMATE_NGPU:-$(nvidia-smi -L 2>/dev/null | wc -l)}
[ "${NGPU:-0}" -ge 1 ] || NGPU=1
KERNEL=${DATAMATE_KERNEL:-python3}
mkdir -p "$OUT"

pids=()
for ((i = 0; i < NGPU; i++)); do
  CUDA_VISIBLE_DEVICES=$i papermill notebooks/02_agent.ipynb "$OUT/02_agent.shard$i.ipynb" \
    --cwd notebooks -k "$KERNEL" --log-output \
    -p SPLIT "$SPLIT" -p OUT "$OUT" -p SHARD "$i/$NGPU" "$@" > "$OUT/shard$i.txt" 2>&1 &
  pids+=($!)
done
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
if [ -f "$OUT/submission.jsonl" ]; then
  echo "$(wc -l < "$OUT/submission.jsonl") answers in $OUT/submission.jsonl"
fi
exit $status
