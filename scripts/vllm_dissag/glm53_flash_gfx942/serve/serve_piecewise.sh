#!/bin/bash
# Runs INSIDE the nightly container. Serve GLM-5.3-Flash-FP8 TP4 with DECODE CUDA
# GRAPH (PIECEWISE) — the validated low-TPOT config.
#
# WINNING CONFIG (2026-09-22, nite@015): cudagraph_mode=PIECEWISE +
# VLLM_USE_BREAKABLE_CUDAGRAPH=1 + patches L1(01-05) + L2(06) + L2c(07).
# Verified correct: "The capital of France is" -> " Paris. In French, Paris is
# spelled \"Paris\", ...". Graph capturing finished; server up.
#
# WHY BREAKABLE=1: GLM-5.3's kpool sparse indexer is NOT capture-safe (its kernels
# do stateful kv/topk writes). Under plain PIECEWISE those writes get frozen into
# the captured graph -> replay = garbage ("locklock"). Breakable makes the
# indexer's @eager_break_during_capture fire, running it OUTSIDE the graph on the
# capture stream with a stable output-buffer address. The custom-op patches (06/07)
# are ALSO required — they let the trace reach capture without a hasattr break.
set -u
export VLLM_USE_V1=1
export VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MLA=1 VLLM_ROCM_USE_AITER_MOE=1 VLLM_ROCM_USE_AITER_RMSNORM=1 VLLM_ROCM_USE_AITER_FP8BMM=0
export VLLM_USE_DEEP_GEMM=0 VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=512
export VLLM_USE_BREAKABLE_CUDAGRAPH=1          # <-- the key that makes decode cudagraph correct
# --- persist JIT caches on host NVMe (kills recompile tax on restart) ---
export AITER_JIT_DIR=/opt/vllm_cache/aiter_jit AITER_META_DIR=/opt/vllm_cache/aiter_meta TRITON_CACHE_DIR=/opt/vllm_cache/triton_cache VLLM_CACHE_ROOT=/opt/vllm_cache/vllm
mkdir -p $AITER_JIT_DIR $AITER_META_DIR $TRITON_CACHE_DIR $VLLM_CACHE_ROOT 2>/dev/null
M=/models/GLM-5.3-Flash-FP8
CC='{"cudagraph_mode":"PIECEWISE"}'
exec vllm serve "$M" -tp 4 --port 20066 --gpu_memory_utilization 0.85 \
  --trust-remote-code --max-model-len 32768 --max-num-seqs 8 --max-num-batched-tokens 8192 --skip-mm-profiling \
  --compilation-config "$CC" \
  2>&1 | tee /opt/vllm_cache/nightly_piecewise.log
