#!/bin/bash
# Runs INSIDE the nightly container. Serve GLM-5.3-Flash-FP8 TP4 EAGER (Phase 0 load test).
set -u
M=/models/GLM-5.3-Flash-FP8
# Persist aiter (~15min GEMM build) + triton JIT + vllm caches on host NVMe
# (/opt/vllm_cache is a host mount) so `docker restart nite` does NOT recompile.
# First serve compiles once; every subsequent restart reuses.
export AITER_JIT_DIR=/opt/vllm_cache/aiter_jit AITER_META_DIR=/opt/vllm_cache/aiter_meta \
  TRITON_CACHE_DIR=/opt/vllm_cache/triton_cache VLLM_CACHE_ROOT=/opt/vllm_cache/vllm
mkdir -p "$AITER_JIT_DIR" "$AITER_META_DIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT" 2>/dev/null
exec vllm serve "$M" -tp 4 --port 20066 --gpu_memory_utilization 0.85 \
  --trust-remote-code --enforce-eager --max-model-len 32768 --max-num-seqs 8 \
  --skip-mm-profiling 2>&1 | tee /opt/vllm_cache/nightly_eager.log
