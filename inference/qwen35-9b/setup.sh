#!/usr/bin/env bash
# One-time setup for the Qwen3.5-9B vLLM server (idempotent, no sudo). Re-running skips finished steps.
#
#   1. uv venv on Python 3.12                         -> $VENV            (default ./.venv-vllm)
#   2. fetch the pinned vLLM source                   -> $VLLM_SRC        (default ./vendor/vllm)
#   3. apply patches/gdn-batch-invariance.patch       (needed for VLLM_BATCH_INVARIANT=1 with Qwen3.5's GDN layers)
#   4. install the pinned packages (requirements-vllm.txt, torch 2.14.0+cu130) and vLLM editable on the
#      precompiled binaries of its upstream base commit (no CUDA compile)
#   5. download the weights (~19 GB) with `hf download` (huggingface-hub 2.x; `huggingface-cli` was renamed `hf`)
#
# Env overrides:
#   VENV, VLLM_SRC                  install locations
#   VLLM_REPO, VLLM_COMMIT          vLLM source (default: the LFM3 PR fork at the commit the profile was measured on)
#   WHEEL_COMMIT, WHEEL_VARIANT     precompiled binaries (default: upstream base fac2ce2, cu130)
#   TORCH_BACKEND                   uv --torch-backend (default cu130; CUDA >= 12.8 wheels are required on Blackwell)
#   MODEL, MODEL_REVISION           weights to download; SKIP_WEIGHTS=1 skips step 5
#   HF_HOME / HF_HUB_CACHE          where weights go (standard Hugging Face variables); HF_TOKEN is not needed
#   FORCE_REINSTALL=1               redo step 4 even if the stamp says it is done
#
# Driver / CUDA are NOT installed here (needs root): see README.md, "Yêu cầu".
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VENV="${VENV:-$HERE/.venv-vllm}"
VLLM_SRC="${VLLM_SRC:-$HERE/vendor/vllm}"
# The tree the profile was measured on: vllm-project/vllm PR #64959 (LFM3) + #64647 on top of upstream fac2ce2.
# Qwen3.5 does not need the two LFM3 commits (they only add lfm3.py and a width-7 conv path), so upstream fac2ce2
# is the fallback when the fork commit is gone.
VLLM_REPO="${VLLM_REPO:-https://github.com/atmikshetty/vllm.git}"
VLLM_COMMIT="${VLLM_COMMIT:-5f67fbcbb2336b0e551c0df2b4c0eb51b030c553}"
FALLBACK_REPO="https://github.com/vllm-project/vllm.git"
FALLBACK_COMMIT="fac2ce2eb3633c44a104a66dbb30d0e6c7f4493e"
# Precompiled CUDA/C++ binaries from wheels.vllm.ai for the upstream base commit. Pinned on purpose: without it,
# setup.py searches from the *current* upstream main, which changes every day.
WHEEL_COMMIT="${WHEEL_COMMIT:-fac2ce2eb3633c44a104a66dbb30d0e6c7f4493e}"
WHEEL_VARIANT="${WHEEL_VARIANT:-cu130}"
TORCH_BACKEND="${TORCH_BACKEND:-cu130}"
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
MODEL_REVISION="${MODEL_REVISION:-c202236235762e1c871ad0ccb60c8ee5ba337b9a}"

LOCK="$HERE/requirements-vllm.txt"
PATCH="$HERE/patches/gdn-batch-invariance.patch"
PY="$VENV/bin/python"
STAMP="$VENV/.qwen35-setup-stamp"

log() { printf '\n==> %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

# --- 0. prerequisites (checked, not installed) ---------------------------------------------------------------
log "0/5 prerequisites"
for c in uv git curl; do command -v "$c" >/dev/null || die "'$c' not found (uv: https://docs.astral.sh/uv/)"; done
[ "$(uname -m)" = "x86_64" ] || echo "WARNING: requirements-vllm.txt was frozen on x86_64; $(uname -m) is untested."
if command -v nvidia-smi >/dev/null && drv="$(nvidia-smi --query-gpu=driver_version,name,memory.total --format=csv,noheader 2>/dev/null)"; then
  echo "GPU: $drv"
  major="${drv%%.*}"
  [[ "$major" =~ ^[0-9]+$ ]] && [ "$major" -lt 580 ] && \
    echo "WARNING: driver $major < 580: the cu130 wheels need an R580+ driver (tested: 595.99)."
else
  echo "WARNING: nvidia-smi does not work. Install the driver first (README.md); continuing with the download steps."
fi

# --- 1. venv ---------------------------------------------------------------------------------------------------
log "1/5 venv $VENV"
if [ -x "$PY" ] && "$PY" -c 'import sys; sys.exit(sys.version_info[:2] != (3, 12))'; then
  echo "exists ($("$PY" -V))"
else
  [ -e "$VENV" ] && die "$VENV exists but is not a Python 3.12 venv; remove it or set VENV=..."
  uv venv --python 3.12 "$VENV"
fi

# --- 2. vLLM source --------------------------------------------------------------------------------------------
log "2/5 vLLM source $VLLM_SRC"
fetch_commit() {  # repo commit -> checks out commit in $VLLM_SRC (shallow fetch by SHA)
  git -C "$VLLM_SRC" remote remove origin 2>/dev/null || true
  git -C "$VLLM_SRC" remote add origin "$1"
  git -C "$VLLM_SRC" fetch --depth 1 origin "$2" && git -C "$VLLM_SRC" checkout -q --detach FETCH_HEAD
}
if [ -d "$VLLM_SRC/.git" ]; then
  head="$(git -C "$VLLM_SRC" rev-parse HEAD)"
  if [ "$head" = "$VLLM_COMMIT" ] || [ "$head" = "$FALLBACK_COMMIT" ]; then
    echo "exists at $head"
  else
    die "$VLLM_SRC is at $head, expected $VLLM_COMMIT; remove it or set VLLM_SRC=..."
  fi
else
  mkdir -p "$VLLM_SRC"
  git -C "$VLLM_SRC" init -q
  if ! fetch_commit "$VLLM_REPO" "$VLLM_COMMIT"; then
    echo "WARNING: could not fetch $VLLM_COMMIT from $VLLM_REPO;"
    echo "         falling back to upstream $FALLBACK_COMMIT (same Qwen3.5 code, without the LFM3 commits)."
    fetch_commit "$FALLBACK_REPO" "$FALLBACK_COMMIT"
  fi
  echo "checked out $(git -C "$VLLM_SRC" rev-parse HEAD)"
fi

# --- 3. patch --------------------------------------------------------------------------------------------------
log "3/5 patch $(basename "$PATCH")"
if git -C "$VLLM_SRC" apply --reverse --check "$PATCH" 2>/dev/null; then
  echo "already applied"
else
  git -C "$VLLM_SRC" apply --check "$PATCH" || die "patch does not apply to $VLLM_SRC (modified tree?)"
  git -C "$VLLM_SRC" apply "$PATCH"
  echo "applied"
fi

# --- 4. packages -----------------------------------------------------------------------------------------------
log "4/5 packages (torch backend $TORCH_BACKEND, vLLM precompiled $WHEEL_COMMIT/$WHEEL_VARIANT)"
want="$(cat "$LOCK" "$PATCH" | sha256sum | cut -c1-16) $(git -C "$VLLM_SRC" rev-parse HEAD) $WHEEL_COMMIT $WHEEL_VARIANT"
if [ "${FORCE_REINSTALL:-0}" != "1" ] && [ -f "$STAMP" ] && [ "$(cat "$STAMP")" = "$want" ] \
   && "$PY" -c 'import vllm, torch' 2>/dev/null; then
  echo "up to date ($("$PY" -c 'import vllm, torch; print("vllm", vllm.__version__, "torch", torch.__version__)'))"
else
  # Every pinned package first (exact versions of the measured venv), then vLLM itself, editable, on the
  # precompiled binaries; -c keeps its resolution on the same pins.
  uv pip install --python "$PY" --torch-backend "$TORCH_BACKEND" -r "$LOCK"
  VLLM_USE_PRECOMPILED=1 \
  VLLM_PRECOMPILED_WHEEL_COMMIT="$WHEEL_COMMIT" \
  VLLM_PRECOMPILED_WHEEL_VARIANT="$WHEEL_VARIANT" \
    uv pip install --python "$PY" --torch-backend "$TORCH_BACKEND" -c "$LOCK" -e "$VLLM_SRC"
  "$PY" -c 'import vllm, torch; print("vllm", vllm.__version__, "torch", torch.__version__, "cuda", torch.version.cuda)'
  echo "$want" >"$STAMP"
fi

# --- 5. weights ------------------------------------------------------------------------------------------------
log "5/5 weights $MODEL@$MODEL_REVISION"
if [ "${SKIP_WEIGHTS:-0}" = "1" ]; then
  echo "skipped (SKIP_WEIGHTS=1)"
else
  # Resumable and skips files already in the HF cache.
  "$VENV/bin/hf" download "$MODEL" --revision "$MODEL_REVISION"
fi

log "done. Start: $HERE/server.sh start   Smoke test: $PY $HERE/smoke_test.py"
