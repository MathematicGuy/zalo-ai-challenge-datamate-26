#!/usr/bin/env bash
# Docker entrypoint (Data tab): read the test set from /data, write /result/submission.jsonl and
# /result/logs/<id>.json. /data is read-only, so the table cache goes to /tmp.
set -euo pipefail
export DATAMATE_CACHE="${DATAMATE_CACHE:-/tmp/datamate_cache}"
export DATAMATE_GPU_UTIL="${DATAMATE_GPU_UTIL:-0.9}"     # the GPUs are ours alone in the container
export DATAMATE_MAX_SEQS="${DATAMATE_MAX_SEQS:-48}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec "$(dirname "$0")/run.sh" "${1:-/data}" "${2:-/result}"
