#!/usr/bin/env python3
# ============================================================================
# GLM53_DISAGG_INDEXER_KV_BARRIER  (patch 21)
#
# THE DISAGG-RECALL FIX for modern EP8/EP8 over MoRIIO (PD-disaggregation).
#
# ROOT CAUSE (measured on 015 prefill / 028 decode, 2026-09-25):
#   Modern disagg recall is INTERMITTENT >~16K (16K = 1/4 PASS, 64K/256K FAIL)
#   while the IDENTICAL colocated config passes 256K deterministically. The DSA
#   sparse indexer's KV reads on the DECODE leg are NOT protected by the PD
#   read-completion barrier:
#     * Standard attention layers are wrapped with @maybe_transfer_kv_layer
#       (model_executor/layers/attention/{attention,mla_attention}.py), whose
#       on-entry connector.wait_for_layer_load(layer_name) BLOCKS until that
#       layer's MoRIIO RDMA READ has landed. -> dense attention is always
#       correct on disagg.
#     * The DSA indexer (this file's sparse_attn_indexer_kpool + the glm5next
#       Indexer module) has NO such wrapper (grep = 0). So on the decode leg it
#       can read the index k_cache AND the kpool tail_cache while their RDMA
#       READ is still in flight -> intermittent stale/torn pooled K -> wrong
#       topk pool selection -> needle intermittently missed. Longer context =
#       more blocks in flight = higher miss rate (16K intermittent -> 64K fail).
#   Proof it is the transfer barrier, not the indexer logic: patch 20 (stable
#   topk) is recall-NEUTRAL here (1/4 both ways); colocated passes with the
#   same C++ topk; INDEXER_SKIP=1 also fails (recompute depends on the equally
#   unbarriered transferred raw K); and the failure is INTERMITTENT on IDENTICAL
#   input (a race, which deterministic topk ties cannot produce).
#
# FIX: at the top of sparse_attn_indexer_kpool (decode/consumer leg only, and
#   only when a KV-transfer connector is active), call
#   connector.wait_for_layer_load(prefix) for BOTH the indexer k_cache prefix
#   and the tail_cache prefix BEFORE any cache read. This is the identical
#   barrier the dense attention layers already use. It runs here in the eager
#   indexer body (the indexer is @eager_break_during_capture / breakable), so
#   it does not get recorded into a captured cudagraph segment. wait_for_layer_load
#   itself no-ops on the producer leg and when not in READ mode, and early-returns
#   under FULL cudagraph capture, so this is safe across roles/modes.
#
# Idempotent, anchor-based, self-skip.  Target (fork decode path):
#   vllm/model_executor/layers/sparse_attn_indexer_kpool.py
# ============================================================================
import io, os, sys

MARKER = "GLM53_DISAGG_INDEXER_KV_BARRIER"
CANDIDATES = [
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer_kpool.py",
]
if len(sys.argv) > 1:
    CANDIDATES.insert(0, sys.argv[1])
F = next((p for p in CANDIDATES if os.path.isfile(p)), None)
if F is None:
    print(f"[{MARKER}] target not found; SKIP"); sys.exit(0)

src = io.open(F, encoding="utf-8").read()
if MARKER in src:
    print(f"[{MARKER}] already applied; SKIP"); sys.exit(0)

# Anchor: the first two lines of the fn body (attn_metadata fetch + resolve).
anchor = (
    "    # careful! this will be None in dummy run\n"
    "    attn_metadata = get_forward_context().attn_metadata\n"
    "    fp8_dtype = current_platform.fp8_dtype()\n"
    "    k_cache_prefix = _resolve_layer_name(k_cache_prefix)\n"
)
inject = (
    "    # careful! this will be None in dummy run\n"
    "    attn_metadata = get_forward_context().attn_metadata\n"
    "    fp8_dtype = current_platform.fp8_dtype()\n"
    "    k_cache_prefix = _resolve_layer_name(k_cache_prefix)\n"
    "\n"
    "    # " + MARKER + ": on the PD DECODE (consumer) leg the indexer's index\n"
    "    # k_cache and kpool tail_cache arrive over MoRIIO RDMA. Unlike the dense\n"
    "    # attention layers (wrapped by @maybe_transfer_kv_layer), the indexer has\n"
    "    # no PD read barrier, so it can read these caches before their transfer\n"
    "    # lands -> intermittent long-context recall garbage. Block until both\n"
    "    # layers' reads have completed, exactly as the dense layers do. No-ops on\n"
    "    # the producer leg / non-READ mode / FULL-cudagraph capture (see\n"
    "    # MoRIIOConnector.wait_for_layer_load). Runs in the eager indexer body.\n"
    "    if isinstance(attn_metadata, dict):\n"
    "        try:\n"
    "            from vllm.distributed.kv_transfer.kv_transfer_state import (\n"
    "                get_kv_transfer_group as _glm53_get_kvt,\n"
    "                has_kv_transfer_group as _glm53_has_kvt,\n"
    "                is_v1_kv_transfer_group as _glm53_is_v1_kvt,\n"
    "            )\n"
    "            if _glm53_has_kvt():\n"
    "                _glm53_conn = _glm53_get_kvt()\n"
    "                if _glm53_is_v1_kvt(_glm53_conn) and _glm53_conn.has_connector_metadata():\n"
    "                    _glm53_conn.wait_for_layer_load(k_cache_prefix)\n"
    "                    if tail_prefix is not None:\n"
    "                        _glm53_conn.wait_for_layer_load(\n"
    "                            _resolve_layer_name(tail_prefix)\n"
    "                        )\n"
    "        except Exception as _glm53_bar_e:  # never brick serving on a barrier hiccup\n"
    "            logger.warning(\n"
    "                '" + MARKER + ": indexer KV barrier skipped (%s)', _glm53_bar_e\n"
    "            )\n"
)
if anchor not in src:
    print(f"[{MARKER}] ANCHOR NOT FOUND (fork body differs) — SKIP"); sys.exit(2)
src = src.replace(anchor, inject, 1)
io.open(F, "w", encoding="utf-8").write(src)
print(f"[{MARKER}] applied to {F}")
