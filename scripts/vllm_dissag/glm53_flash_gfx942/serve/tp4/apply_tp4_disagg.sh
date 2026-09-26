#!/bin/bash
# Apply the FULL GLM-5.3-Flash disagg stack inside nite (idempotent).
# Expects staged at: /tmp/tp4stage/{core/*.py, slotfix.patch, moriio_kbpb_fix/*.py}
# Does: core 01-07,11 -> slotfix(patch14 recall) -> kbpb moriio overlay -> clear pycache -> verify.
set -u
S=/tmp/tp4stage
D=/usr/local/lib/python3.12/dist-packages/vllm
MORIIO="$D/distributed/kv_transfer/kv_connector/v1/moriio"

echo "=== CORE patches (01-07,11) ==="
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
  python3 "$S/core/$f" || echo "  (warn: $f returned nonzero)"
done

echo "=== slotfix / patch 14 (recall correctness) ==="
if grep -q GLM53_KPOOL_SLOT_MAPPING_FIX "$D/v1/attention/backends/mla/indexer.py" 2>/dev/null; then
  echo "  slotfix already present ($(grep -c GLM53_KPOOL_SLOT_MAPPING_FIX $D/v1/attention/backends/mla/indexer.py) markers)"
else
  cd /usr/local/lib/python3.12/dist-packages && patch -p1 --forward < "$S/slotfix.patch" && echo "  slotfix APPLIED" || echo "  slotfix FAILED rc=$?"
fi

echo "=== kbpb MoRIIO overlay (moriio_layout.py + moriio_connector.py + common + engine) ==="
if [ -d "$MORIIO" ]; then
  cp "$S/moriio_kbpb_fix/moriio_layout.py"    "$MORIIO/moriio_layout.py"
  cp "$S/moriio_kbpb_fix/moriio_connector.py" "$MORIIO/moriio_connector.py"
  cp "$S/moriio_kbpb_fix/moriio_common.py"    "$MORIIO/moriio_common.py"
  cp "$S/moriio_kbpb_fix/moriio_engine.py"    "$MORIIO/moriio_engine.py"
  rm -rf "$MORIIO/__pycache__"
  echo "  overlaid + cleared pycache"
else
  echo "  ERROR: moriio dir not found at $MORIIO"
  find "$D" -path '*kv_connector/v1/moriio' -type d 2>/dev/null | head
fi

# stage the NIAH harness for later
cp "$S/moriio_kbpb_fix/niah_disagg.py" /opt/vllm_cache/niah_disagg.py 2>/dev/null
cp "$S/moriio_kbpb_fix/niah_grid.sh"   /opt/vllm_cache/niah_grid.sh   2>/dev/null

echo "=== VERIFY markers ==="
echo "slotfix   = $(grep -c GLM53_KPOOL_SLOT_MAPPING_FIX $D/v1/attention/backends/mla/indexer.py 2>/dev/null)"
echo "dispatch  = $(grep -c GLM53_AMD_INDEXER_DISPATCH $D/models/glm5next/amd/sparse_indexer.py 2>/dev/null)"
echo "kpool06   = $(grep -c GLM53_KPOOL_CUSTOM_OP $D/models/glm5next/amd/sparse_indexer.py 2>/dev/null)"
echo "kbpb      = $(grep -c GLM53_INDEXER_KBPB $MORIIO/moriio_layout.py 2>/dev/null)"
echo "hma       = $(grep -c GLM53_RELAX_HMA_GUARD $MORIIO/moriio_connector.py 2>/dev/null)"
echo "pergroup  = $(grep -c GLM53_PERGROUP_REMOTE_BLOCKS $MORIIO/moriio_connector.py 2>/dev/null)"
echo "packed4d  = $(grep -c GLM53_FLASH_PACKED_4D_FALLBACK $MORIIO/moriio_layout.py 2>/dev/null)"
echo "=== py_compile sanity ==="
python3 -c "import py_compile,sys; [py_compile.compile(f, doraise=True) for f in ['$MORIIO/moriio_layout.py','$MORIIO/moriio_connector.py','$MORIIO/moriio_common.py','$MORIIO/moriio_engine.py']]; print('  moriio overlay compiles OK')" || echo "  COMPILE ERROR in overlay"
echo "=== DONE ==="
