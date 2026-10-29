#!/usr/bin/env bash
# Docker entrypoint (Data tab): read the test set from /data, write /result/submission.jsonl and /result/logs/.
# /data is read-only, so the table cache goes to /tmp.
set -euo pipefail
DATA="${1:-/data}"
OUT="${2:-/result}"
export DATAMATE_CACHE="${DATAMATE_CACHE:-/tmp/datamate_cache}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p "$OUT/logs"
python -m datamate.predict --data "$DATA" --out "$OUT"
