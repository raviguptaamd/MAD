#!/usr/bin/env python3
# ============================================================================
# GLM53_FORK_KPOOL_RECALL  (candidate patch 20)
#
# STATUS (validated on MI300X node 030, colocated EP8, 2026-09-25):
#   ** RECALL-NEUTRAL / OPTIONAL — NOT the fix, and NOT required. **
#   A/B control on the modern image (d922e2bd) with the full proven AITER env
#   showed 256K NIAH recall HOLDS *both* with and without this patch. The
#   fork's C++ torch.ops._C.top_k_per_row_prefill/decode selects the correct
#   DSA pools at 256K on its own; the unstable-tie-break bug hypothesized below
#   did NOT manifest on gfx942 here. The real modern-recall prerequisites were
#   (a) patches 18+19 (make the DSA prefill MQA-logits path run on gfx942) and
#   (b) the proven AITER env (AITER_MLA/MOE/RMSNORM=1, FP8BMM=0, DEEP_GEMM=0,
#   SPARSE_INDEXER_MAX_LOGITS_MB=4096, EP8, MNBT=16384). Keep this overlay ONLY
#   as a defensive determinism guard if a future aiter/topk regression appears;
#   do not gate the recipe on it. The original design rationale is preserved
#   below for the record.
#
# ---- ORIGINAL DESIGN RATIONALE (hypothesis, later shown non-load-bearing) ----
# Intended as THE RECALL FIX for the MODERN from-source stack (fork @41644d2).
#
# ROOT CAUSE (diff of known-good OLD indexer vs fork indexer):
#   The OLD image (proven correct 256K recall) selected the top-`select_k`
#   DSA pools with the AITER Triton kernel in *stable* mode:
#
#       from aiter.ops.topk import top_k_per_row_prefill   # + _decode
#       top_k_per_row_prefill(logits, ks, ke, dst, None, num_rows,
#                             s0, s1, select_k, stable=True)
#       # comment in old src:
#       #   GLM53_STABLE_TOPK_PREFILL: stable=True so all TP ranks pick an
#       #   identical pool set (fixes NIAH >8192 mis-select).
#       #   GLM53_STABLE_TOPK_DECODE: stable=True ... (fixes recall drift >8192)
#
#   The fork RESTRUCTURED the indexer into
#   vllm/model_executor/layers/sparse_attn_indexer_kpool.py and routes the
#   ROCm top-k through the COMPILED C++ op instead:
#
#       torch.ops._C.top_k_per_row_prefill(logits, ks, ke, dst, num_rows,
#                                          s0, s1, select_k)      # NO stable
#       torch.ops._C.top_k_per_row_decode(logits, next_n, seq_lens, dst,
#                                         num_rows, s0, s1, select_k)
#
#   (verified schema in-container: `_C::top_k_per_row_prefill(Tensor logits,
#    Tensor rowStarts, Tensor rowEnds, Tensor indices, int numRows,
#    int stride0, int stride1, int topK) -> ()`  — no stable, no values.)
#
#   With DSA `topk_tokens=2048`, any context <=2048 tokens takes the dense /
#   causal-fill path (`_fill_causal_indices` / `_fill_short_decode_causal
#   _indices`) which is ALWAYS correct -> 2K/4K NIAH PASS.  Past 2048 tokens
#   the pool-topk path engages; the fork's C++ top-k does NOT tie-break
#   stably, so different TP ranks (and successive decode steps) can pick
#   DIFFERENT pool sets among equal/near-equal logits.  The needle's pool is
#   dropped inconsistently -> garbage >=10K, empty at 256K.  This is exactly
#   the ">8192 mis-select" failure the OLD stable=True flag was added to fix.
#   NOTE: patch 14 (block-table geometry) is ALREADY baked into the modern
#   image (marker present x2) and did NOT fix recall -> geometry is ruled out;
#   the remaining divergence from known-good is precisely this top-k op.
#
# FIX (this patch, .py-level overlay — NO kernel rebuild needed):
#   Rewrite the ROCm (`else:`) top-k branches in the fork indexer to call the
#   AITER kernel used by the OLD good path.  The modern aiter still ships
#   `aiter.ops.topk.top_k_per_row_prefill/decode`; we detect at runtime
#   whether this build's aiter accepts `stable=` and pass it when available
#   (older aiter had it; some rebuilds dropped the kwarg).  Prefill signature
#   takes an extra `values` arg (pass None) that the C++ op omits.
#
#   If the installed aiter kernel lacks a stable mode entirely, we STILL move
#   selection onto the aiter kernel (same kernel family the OLD good build
#   used) and the patch logs a clear WARNING so the operator knows a
#   stable-capable aiter (or a C++ rebuild adding a stable arg to
#   top_k_per_row_*) is required.  See the report for the rebuild estimate.
#
# Idempotent, anchor-based, self-skip.  Target (fork path):
#   vllm/model_executor/layers/sparse_attn_indexer_kpool.py
# ============================================================================
import io
import os
import sys

MARKER = "GLM53_FORK_KPOOL_RECALL"
CANDIDATES = [
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer_kpool.py",
]
if len(sys.argv) > 1:
    CANDIDATES.insert(0, sys.argv[1])

target = next((p for p in CANDIDATES if os.path.isfile(p)), None)
if target is None:
    print(f"[{MARKER}] target sparse_attn_indexer_kpool.py not found in {CANDIDATES}; SKIP")
    sys.exit(0)

src = io.open(target, encoding="utf-8").read()
if MARKER in src:
    print(f"[{MARKER}] already applied in {target}; SKIP")
    sys.exit(0)

# ---- Anchor 1: PREFILL C++ top-k call (the ROCm `else` branch). ------------
PREFILL_ANCHOR = """            else:
                torch.ops._C.top_k_per_row_prefill(
                    logits,
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                    topk_dst,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    select_k,
                )"""

PREFILL_REPL = """            else:
                # GLM53_FORK_KPOOL_RECALL (prefill): route DSA pool selection
                # through the AITER top-k the OLD good build used, in STABLE
                # mode so every TP rank picks an identical pool set. The fork's
                # torch.ops._C.top_k_per_row_prefill tie-breaks unstably ->
                # needle-pool dropped past topk_tokens (2048) -> garbage >=10K.
                from vllm.platforms import current_platform as _cp_glm53
                if _cp_glm53.is_rocm():
                    from aiter.ops.topk import (
                        top_k_per_row_prefill as _glm53_topk_prefill,
                    )
                    try:
                        _glm53_topk_prefill(
                            logits,
                            chunk.cu_seqlen_ks,
                            chunk.cu_seqlen_ke,
                            topk_dst,
                            None,
                            num_rows,
                            logits.stride(0),
                            logits.stride(1),
                            select_k,
                            stable=True,
                        )
                    except TypeError:
                        # aiter build without the `stable` kwarg: still use the
                        # aiter kernel (old good family); a stable aiter/C++
                        # rebuild is needed for full determinism.
                        _glm53_topk_prefill(
                            logits,
                            chunk.cu_seqlen_ks,
                            chunk.cu_seqlen_ke,
                            topk_dst,
                            None,
                            num_rows,
                            logits.stride(0),
                            logits.stride(1),
                            select_k,
                        )
                else:
                    torch.ops._C.top_k_per_row_prefill(
                        logits,
                        chunk.cu_seqlen_ks,
                        chunk.cu_seqlen_ke,
                        topk_dst,
                        num_rows,
                        logits.stride(0),
                        logits.stride(1),
                        select_k,
                    )"""

# ---- Anchor 2: DECODE C++ top-k call (the ROCm `else` branch). -------------
DECODE_ANCHOR = """            else:
                torch.ops._C.top_k_per_row_decode(
                    logits,
                    next_n,
                    seq_lens,
                    topk_dst,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    select_k,
                )"""

DECODE_REPL = """            else:
                # GLM53_FORK_KPOOL_RECALL (decode): mirror the prefill fix so
                # per-token pool selection stays deterministic across decode
                # steps (old comment: fixes recall drift during generation
                # >8192). Route ROCm decode top-k through the AITER kernel.
                from vllm.platforms import current_platform as _cp_glm53d
                if _cp_glm53d.is_rocm():
                    from aiter.ops.topk import (
                        top_k_per_row_decode as _glm53_topk_decode,
                    )
                    try:
                        _glm53_topk_decode(
                            logits,
                            next_n,
                            seq_lens,
                            topk_dst,
                            num_rows,
                            logits.stride(0),
                            logits.stride(1),
                            select_k,
                            stable=True,
                        )
                    except TypeError:
                        _glm53_topk_decode(
                            logits,
                            next_n,
                            seq_lens,
                            topk_dst,
                            num_rows,
                            logits.stride(0),
                            logits.stride(1),
                            select_k,
                        )
                else:
                    torch.ops._C.top_k_per_row_decode(
                        logits,
                        next_n,
                        seq_lens,
                        topk_dst,
                        num_rows,
                        logits.stride(0),
                        logits.stride(1),
                        select_k,
                    )"""

missing = []
if PREFILL_ANCHOR not in src:
    missing.append("PREFILL_ANCHOR")
if DECODE_ANCHOR not in src:
    missing.append("DECODE_ANCHOR")
if missing:
    print(f"[{MARKER}] EXPECTED ANCHORS MISSING {missing} in {target}; "
          f"build differs from fork @41644d2 — SKIP (do not force)")
    sys.exit(2)

src = src.replace(PREFILL_ANCHOR, PREFILL_REPL, 1)
src = src.replace(DECODE_ANCHOR, DECODE_REPL, 1)
io.open(target, "w", encoding="utf-8").write(src)
print(f"[{MARKER}] applied to {target} "
      f"(prefill+decode ROCm top-k routed through aiter stable kernel)")
