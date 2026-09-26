#!/usr/bin/env python3
# MODERN-STACK FIX (task: modern from-source image bring-up, 2026-09-25)
#
# BUG (measured on the modern image, EP8/EP8 prefill, first request):
#   RuntimeError: Worker failed with error 'cannot import name 'flydsl_fp8_mqa_logits'
#   from 'aiter.ops.flydsl'
#
# ROOT CAUSE: the glm5next vLLM fork (raviguptaamd/vllm @41644d2) DSA single-seq
#   prefill path rocm_fp8_mqa_logits() has a gfx942 fast-branch that does a HARD
#   `from aiter.ops.flydsl import flydsl_fp8_mqa_logits`. That symbol existed in the
#   OLDER aiter the fork was written against, but TIP aiter (raviguptaamd/aiter
#   @b50066a9, flydsl 0.2.2) DROPPED it — aiter.ops.flydsl now exports
#   flydsl_flash_attn_func / flydsl_hgemm / flydsl_moe_* / flydsl_qk_norm_rope_quant
#   (no flydsl_fp8_mqa_logits). So the gfx942 prefill branch dies on EVERY >0 prompt.
#   (Our patch 11 targeted aiter/ops/flydsl/kernels/mqa_logits/fp8_mqa_logits.py which
#   ALSO no longer exists on tip aiter -> patch 11 is a no-op on the modern stack.)
#
# FIX: guard the flydsl fast-path import. If flydsl_fp8_mqa_logits is importable, use
#   it (fast path preserved on an aiter that has it). If NOT, fall through to the
#   EXISTING triton fallback right below it (mqa_logits_module().fp8_mqa_logits ->
#   aiter.ops.triton.attention.fp8_mqa_logits, which tip aiter DOES ship, same
#   signature). Behavior-preserving where flydsl exists; correct + non-fatal on tip.
#
# Marker: GLM53_FLYDSL_FP8_MQA_IMPORT_GUARD
# Target: vllm/v1/attention/ops/rocm_aiter_mla_sparse.py
import os, sys

MARKER = "GLM53_FLYDSL_FP8_MQA_IMPORT_GUARD"
CANDIDATES = [
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py",
]
if len(sys.argv) > 1:
    CANDIDATES.insert(0, sys.argv[1])
F = next((p for p in CANDIDATES if os.path.isfile(p)), None)
if F is None:
    print(f"[{MARKER}] target not found; SKIP"); sys.exit(0)

src = open(F).read()
if MARKER in src:
    print(f"[{MARKER}] already applied; SKIP"); sys.exit(0)

# The exact fast-path block in rocm_fp8_mqa_logits().
anchor = (
    "    if _ON_GFX942 and rocm_aiter_ops.is_enabled():\n"
    "        from aiter.ops.flydsl import flydsl_fp8_mqa_logits\n"
    "\n"
    "        return flydsl_fp8_mqa_logits(\n"
    "            q, k_fp8, scale, weights, cu_seqlen_ks, cu_seqlen_ke\n"
    "        )\n"
)
replacement = (
    "    if _ON_GFX942 and rocm_aiter_ops.is_enabled():\n"
    "        # " + MARKER + ": tip aiter dropped flydsl_fp8_mqa_logits; use it if\n"
    "        # present, else fall through to the triton fp8_mqa_logits path below.\n"
    "        try:\n"
    "            from aiter.ops.flydsl import flydsl_fp8_mqa_logits\n"
    "        except ImportError:\n"
    "            flydsl_fp8_mqa_logits = None\n"
    "        if flydsl_fp8_mqa_logits is not None:\n"
    "            return flydsl_fp8_mqa_logits(\n"
    "                q, k_fp8, scale, weights, cu_seqlen_ks, cu_seqlen_ke\n"
    "            )\n"
)

if anchor not in src:
    print(f"[{MARKER}] anchor not found (fork call site differs) — SKIP"); sys.exit(0)

src = src.replace(anchor, replacement, 1)
open(F, "w").write(src)
print(f"[{MARKER}] applied to {F}")
