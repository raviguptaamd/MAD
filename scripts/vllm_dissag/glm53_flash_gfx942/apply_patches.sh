#!/bin/bash
# Apply the GLM-5.3-Flash gfx942 patches INSIDE a vllm-nightly container (pinned f169e8df).
# Usage (from host):
#   docker cp glm53_recipe/patches <container>:/tmp/patches
#   docker cp glm53_recipe/apply_patches.sh <container>:/tmp/
#   docker exec <container> bash /tmp/apply_patches.sh
# Idempotent: every patch self-skips if already applied. See BASELINE.md for status of each.
set -u
P=${1:-/tmp/patches}

echo "=== CORE patches (proven; decode cudagraph + fast flydsl path) from $P/core ==="
for f in \
  01_amd_indexer_dispatch.py \
  02_rocm_topk_ready.py \
  03_tilelang_warmup.py \
  04_torch_compile.py \
  05_gdn_hasattr.py \
  06_kpool_custom_op.py \
  07_fused_qk_rmsnorm_op.py \
  11_flydsl_shrui_fix.py ; do
  echo "--- $f ---"
  python3 "$P/core/$f" || echo "  (warn: $f returned nonzero)"
done

# PROVISIONAL: applied by default for now (10 stable-topk is harmless; 12 fnuz-Q is a
# candidate for removal — proven irrelevant to recall). 09 bypass is NOT applied (superseded
# by core/11 flydsl fix; it is a fallback only — do not combine with 11). To include the
# provisional set, pass a 2nd arg: `bash apply_patches.sh /tmp/patches with-provisional`.
if [ "${2:-}" = "with-provisional" ]; then
  echo "=== PROVISIONAL patches from $P/provisional ==="
  for f in 10_stable_topk.py 12_fnuz_q.py ; do
    echo "--- $f ---"
    python3 "$P/provisional/$f" || echo "  (warn: $f returned nonzero)"
  done
fi

echo ""
echo "=== DONE. Serve with serve/serve_piecewise.sh (VLLM_USE_BREAKABLE_CUDAGRAPH=1 +"
echo "    cudagraph_mode PIECEWISE + --max-num-batched-tokens 8192 + --speculative-config mtp). ==="
echo "=== OPEN: long-context recall >~2K (nondeterministic pool selection). See BASELINE.md. ==="
