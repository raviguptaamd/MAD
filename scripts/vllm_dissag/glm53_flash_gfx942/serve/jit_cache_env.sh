#!/bin/bash
# Source this BEFORE `vllm serve` to persist all JIT caches on the host-mounted
# NVMe (/opt/vllm_cache is bind-mounted to host /mnt/.../glm53_cache/nightly_<node>).
# Without this, aiter's ~15-min GEMM build + triton's ~65M kernel cache rebuild on
# EVERY container restart. With it, first run compiles once; every later run reuses.
# AITER_JIT_DIR override REMOVED: pointing aiter at an empty host dir breaks its baked-in build (module_aiter_core not found). Keep aiter default; persist only triton.

export TRITON_CACHE_DIR=/opt/vllm_cache/triton_cache
export VLLM_CACHE_ROOT=/opt/vllm_cache/vllm
export CCACHE_DIR=/opt/vllm_cache/ccache
export FLASHINFER_CACHE_DIR=/opt/vllm_cache/flashinfer 2>/dev/null || true
mkdir -p "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT" "$CCACHE_DIR" 2>/dev/null
