#!/usr/bin/env bash
# Serve Qwen/Qwen3.5-9B with vLLM (OpenAI-compatible API on :$PORT, Prometheus on :$PORT/metrics) in the
# foreground. For a background server with a pidfile use ./server.sh start|stop|status. Setup: ./setup.sh.
#
# Self-contained copy of the tuned workspace launcher (serve_qwen35.sh + the shared serve_lfm3.sh), same flags and
# defaults. Every default can be overridden by an env var; extra CLI args are passed through to `vllm serve`.
# DRY_RUN=1 prints the command instead of starting vLLM (no GPU needed).
#
# Model: Qwen3.5-9B (9.65B, Apache-2.0, released 2026-02-27: eligible for the competition).
# - BF16 is ~19 GB -> online FP8 weights on the 16 GB card.
#   No official Qwen3.5-9B-FP8 checkpoint exists (HF API 2026-11-03: Qwen ships FP8 only for 27B/35B-A3B/122B/397B).
# - Qwen tool calls: qwen3_coder XML; thinking via enable_thinking (client side); reasoning parser qwen3.
# - Text only (--language-model-only: vision tower not loaded).
# - torch.compile is not supported for this model in this build -> CUDA graphs FULL_DECODE_ONLY, prefill eager.
#
# Tuned 2026-11-03 on an RTX 5060 Ti 16 GB (sm_120), DataMate tool mode, thinking on (tuning set A = 36 train
# questions, seed 42; set B = seed 7; disjoint from the evaluation ids):
#                                       s/q 8 workers (cold/warm)  p95      1 worker  A        B        records rerun
#   before (BI off, MTP k=2)            1.04 / 1.36                17 s     4.49      31/4/1   36/0/0   33/36 (Pass 31-34)
#   after  (these defaults)             1.03 / 0.91                19 s     4.81(k2)  32/2/2   35/1/0*  36/36, 1=8=16 workers
#   (* B measured with k=2; MTP k=2/3/4 give identical records on A.) 16 workers = no gain (0.99 s/q): KV-bound.
#   Re-check after a GPU-fault reboot: same defaults reproduce the records 36/36;
#   + KV_CACHE_MEMORY_BYTES 3481M and the client prefix warm-up + per-run cache salt (Zalo-Track-2 llm/config.py):
#   A 0.98 / 0.98 s/q on two consecutive runs after a restart, 32/2/2, records and tool calls 36/36; B 0.91, 35/1/0.
#   1 = 8 = 16 workers records 36/36. Thinking off (LOCAL_LLM_TOOL_THINKING=0): 0.48 s/q, 33/1/2.
# Rejected (set A, 8 workers): 4 workers 1.37 s/q; MTP k=1 1.30-1.51 s/q (+ 2 records changed);
#   --attention-backend TRITON_ATTN 0.89 s/q warm and --kv-cache-dtype fp8 (forces TRITON_ATTN, FA2 on sm_120 has no
#   FP8 KV) 0.76-0.78 s/q (A 32/3/1, B 36/0/0) -- 1.2x faster, but with Triton attention the records change with
#   the worker count (1 vs 8 vs 16: 1 record of 36 flips) while FLASH_ATTN keeps them equal: not adopted (rerun rule).
#   --max-model-len 12288: no memory effect (chunked prefill -> 2048-token batches on this card; the KV pool is set by
#   bytes); it only clamps max_tokens to ~1.8k (a hidden thinking budget; max output seen 706). Not a speed lever.
#
# Batch invariance (BATCH_INVARIANT=1 -> VLLM_BATCH_INVARIANT=1): needs patches/gdn-batch-invariance.patch (applied by
# setup.sh to vendor/vllm): the GDN backend declares batch invariance and, in this mode, the GDN MTP decode runs on the
# Triton path in every step (the fused CUDA MTP kernel only ran in prefill-free steps, so a request's tokens took a
# different kernel whenever another request was prefilling). Without the patch vLLM aborts:
# "VLLM batch_invariant mode is not supported for GDN_ATTN". Measured: BI off 33-35/36 records agree across
# reruns / 1 / 8 / 16 workers; BI on 36/36 everywhere, same speed (38.5 vs 40 tok/s per stream at 8 workers).
# Caveat: SPEC_CONFIG= (no spec) is NOT fully invariant (records 36/36 but tool calls 34/36 rerun): decode-only
# steps use the packed recurrent kernel, mixed steps another one (VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE; untested).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VENV="${VENV:-$HERE/.venv-vllm}"
VLLM="${VLLM:-$VENV/bin/vllm}"

MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
# HF commit the profile was measured on (set empty to follow `main`).
MODEL_REVISION="${MODEL_REVISION-c202236235762e1c871ad0ccb60c8ee5ba337b9a}"
SERVED_NAME="${SERVED_NAME:-qwen3.5-9b}"        # = LOCAL_LLM_MODEL in Zalo-Track-2 llm/config.py
PORT="${PORT:-8000}"
HOST="${HOST:-}"                                # empty = vLLM default (all interfaces)
# Tool-mode prompt is ~10.45k tokens (system + 41 tool schemas + dialect lines) + up to 4k output.
# DataMate prompts (README + schema + views + examples) run to ~10-20k tokens; thinking samples add up to 12k.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
# GPU_MEMORY_UTILIZATION (GPU_MEM_UTIL accepted as the old name). Only sizes the KV pool when
# KV_CACHE_MEMORY_BYTES is empty; vLLM refuses to start when free memory < util x total.
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-${GPU_MEM_UTIL:-0.90}}"
TOOL_PARSER="${TOOL_PARSER:-qwen3_coder}"
REASONING_PARSER="${REASONING_PARSER-qwen3}"    # set empty to disable
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"     # Qwen3.5 is native in this vLLM build
SEED="${SEED:-42}"                              # competition rule: re-run must agree within 0.03 PA

# Batch-invariant kernels: greedy output no longer depends on how many requests share a batch, so the
# DataMate batch runner can send questions in parallel (`--workers 8`) and still reproduce its answers.
# Needs the GDN patch (see header). Set BATCH_INVARIANT=0 to opt out (then reruns drift: 33-35/36 records).
BATCH_INVARIANT="${BATCH_INVARIANT:-1}"
if [ "$BATCH_INVARIANT" = "1" ]; then export VLLM_BATCH_INVARIANT=1; else unset VLLM_BATCH_INVARIANT; fi
# VLLM_USE_BREAKABLE_CUDAGRAPH=0: breakable graphs crashed at capture on Gemma 4 (BI auto-enables them).
export VLLM_USE_BREAKABLE_CUDAGRAPH="${VLLM_USE_BREAKABLE_CUDAGRAPH:-0}"

# Online FP8 weights (per-tensor weights, per-token activation scales: row-independent, so batch invariant).
# QUANT= serves BF16 (needs ~19 GB for weights; under BI, BF16 linears run on slower Triton persistent matmuls).
QUANT="${QUANT-fp8_per_tensor}"
# Speculative decoding: built-in 1-layer MTP head (SPEC_CONFIG= disables). Lossless: records 36/36 identical for
# k=2/3/4. Set A, 8 workers, s/q cold / warm: none 1.48 / 1.25 (30 tok/s/stream) | k2 1.08 / 0.98 (86% acceptance,
# 38.5 tok/s) | k3 1.03 / 0.91 (79%, 41.7) | k4 1.06 / 0.94 (73%, 45.5; KV 55k -> 48k tokens, queueing 0.6-1.3 s).
# k=3: the knee (k=4 decodes faster per stream but the larger draft reserve shrinks KV and requests queue).
DEFAULT_SPEC='{"method": "mtp", "num_speculative_tokens": 3}'
SPEC_CONFIG="${SPEC_CONFIG-$DEFAULT_SPEC}"
# Memory (FP8 load 10.51 GiB: token embedding + untied lm_head stay BF16, ~2 GiB each; KV 2.9 GiB = 51k tokens
# at k=3 with util 0.90, ~5 concurrent 10.4k-token tool prompts). The CUDA-graph memory profiler allocates one KV
# block per sequence and a GDN block holds a whole recurrent state (~21 MB), so the default 256 sequences -> 5.06 GiB
# -> OOM at startup. MAX_NUM_SEQS=16 (keep <= 32). Client: --workers 8 (16 workers: queue 2.7-2.8 s, no gain;
# 4 workers 1.37 s/q).
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
# KV_CACHE_MEMORY_BYTES=3481M (3.40 GiB, replaces the 0.90-utilisation sizing): KV 51,243 -> 59,609 tokens (+16%),
# same records, ~15.0 of 16.3 GB used with a desktop session (vLLM's own hint: 3.68 GiB = "fully utilize"). Set it
# empty to fall back to GPU_MEMORY_UTILIZATION (do that on any card other than 16 GB). Binary suffixes take integers
# only ("3.4G" is rejected at parse time).
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES-3481M}"
# Attention backend: empty = vLLM's choice (FLASH_ATTN on sm_120, the batch-invariant one). TRITON_ATTN is faster
# here but breaks worker-count invariance (see header). This build ignores the VLLM_ATTENTION_BACKEND env var.
ATTENTION_BACKEND="${ATTENTION_BACKEND:-}"

args=(serve "$MODEL"
  --served-model-name "$SERVED_NAME"
  --port "$PORT"
  --dtype bfloat16
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --seed "$SEED"
  --enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER"
  # The checkpoint's generation_config may carry sampling defaults (e.g. a repetition penalty) that change greedy
  # output; 'vllm' = use vLLM defaults so the harness's request params (temperature 0, seed 42) rule.
  --generation-config vllm
  # JSON mode: forbid free whitespace. Measured 2026-11-03 (LFM3): the model wrote an unescaped `"` inside the
  # harness's "reasoning" string, the grammar treated it as the string's end, and only whitespace/`,`/`}` stayed
  # legal -- so the model emitted whitespace until max_tokens (~38 s per call).
  --structured-outputs-config '{"backend": "xgrammar", "disable_any_whitespace": true}'
  # Already on by default in this build, made explicit because the tool-mode workload depends on them:
  # every request repeats the same ~10k-token system + 41 tool schemas prefix -> 89% (cold) / 94% (warm)
  # prefix-cache hits, prefill 3.5-4% of request time (measured 2026-11-03).
  --enable-prefix-caching --enable-chunked-prefill
)
[ -n "$MODEL_REVISION" ] && args+=(--revision "$MODEL_REVISION")
[ -n "$HOST" ] && args+=(--host "$HOST")
[ -n "$SPEC_CONFIG" ] && args+=(--speculative-config "$SPEC_CONFIG")
[ -n "$REASONING_PARSER" ] && args+=(--reasoning-parser "$REASONING_PARSER")
[ "$TRUST_REMOTE_CODE" = "1" ] && args+=(--trust-remote-code)
args+=(--language-model-only --max-num-seqs "$MAX_NUM_SEQS")
[ -n "$KV_CACHE_MEMORY_BYTES" ] && args+=(--kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES")
[ -n "$QUANT" ] && args+=(--quantization "$QUANT")
[ -n "$ATTENTION_BACKEND" ] && args+=(--attention-backend "$ATTENTION_BACKEND")

# DataMate (this repo): serve the SQL LoRA adapter next to the base model. Requests pick it with model="$LORA_NAME";
# model="$SERVED_NAME" stays the plain base model (datamate uses both). LORA= (empty) serves the base model only.
REPO="$(cd "$HERE/../.." && pwd)"
LORA="${LORA-$REPO/adapters/sql_rft_v3}"
LORA_NAME="${LORA_NAME:-sql}"
[ -n "$LORA" ] && args+=(--enable-lora --lora-modules "$LORA_NAME=$LORA" --max-lora-rank "${MAX_LORA_RANK:-64}" --max-loras 1)
# EXTRA_ARGS: any further vLLM flags, word-split (e.g. EXTRA_ARGS="--max-num-batched-tokens 8192").
# shellcheck disable=SC2206
[ -n "${EXTRA_ARGS:-}" ] && args+=(${EXTRA_ARGS})

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "VLLM_BATCH_INVARIANT=${VLLM_BATCH_INVARIANT:-} VLLM_USE_BREAKABLE_CUDAGRAPH=$VLLM_USE_BREAKABLE_CUDAGRAPH"
  printf '%q ' "$VLLM" "${args[@]}" "$@"; echo
  exit 0
fi
if [ ! -x "$VLLM" ]; then
  echo "vllm not found at $VLLM -- run ./setup.sh first (or set VENV=/path/to/venv)." >&2
  exit 1
fi
exec "$VLLM" "${args[@]}" "$@"
