#!/usr/bin/env python3
# MODERN-STACK FIX (task: modern from-source image bring-up, 2026-09-25)
#
# BUG (measured, EP8/EP8 prefill first request, AFTER patch 18 routed past the
# missing flydsl symbol):
#   RuntimeError: out of resource: shared memory, Required: 81920, Hardware limit:
#   65536. Reducing block sizes or `num_stages` may help.
#
# ROOT CAUSE: the DSA sparse indexer prefill MQA-logits path now lands in tip
#   aiter's triton kernel aiter/ops/triton/attention/fp8_mqa_logits.py. That file
#   has a fast "gluon" kernel ONLY for gfx950 and gfx1250 (see its import block:
#   _gluon_kernels/gfx950 and /gfx1250 — NO gfx942). On gfx942 use_gluon is False
#   and it launches the generic _fp8_mqa_logits_kernel with BLOCK_KV=128,
#   num_stages=2 -> 81920 bytes LDS, which EXCEEDS gfx942's 65536-byte LDS limit.
#   (flydsl_fp8_mqa_logits, the fork's original gfx942 fast path, was DELETED from
#   tip aiter — the whole flydsl mqa_logits kernel dir is gone — so it can't be used.)
#
# FIX: in the non-gluon launch path, halve BLOCK_KV (128 -> 64) so the kernel's LDS
#   fits gfx942's 64KB. block_kv only sets the KV tile the kernel stages in shared
#   memory; halving it keeps results identical (it just iterates more KV tiles) and
#   drops LDS to ~40-48KB. Scoped to the non-gluon path (gfx950/gfx1250 keep gluon).
#
# Marker: GLM53_FP8_MQA_GFX942_LDS
# Target: aiter/ops/triton/attention/fp8_mqa_logits.py
import os, sys

MARKER = "GLM53_FP8_MQA_GFX942_LDS"
CANDIDATES = [
    "/usr/local/lib/python3.12/dist-packages/aiter/ops/triton/attention/fp8_mqa_logits.py",
]
if len(sys.argv) > 1:
    CANDIDATES.insert(0, sys.argv[1])
F = next((p for p in CANDIDATES if os.path.isfile(p)), None)
if F is None:
    print(f"[{MARKER}] target not found; SKIP"); sys.exit(0)

src = open(F).read()
if MARKER in src:
    print(f"[{MARKER}] already applied; SKIP"); sys.exit(0)

# The non-gluon path sets block_kv = 128 just before the _fp8_mqa_logits_kernel launch.
anchor = "    if not use_gluon:\n        block_kv = 128\n"
replacement = (
    "    if not use_gluon:\n"
    "        # " + MARKER + ": gfx942 has no gluon fp8_mqa_logits kernel; the generic\n"
    "        # kernel at BLOCK_KV=128 needs 80KB LDS > gfx942's 64KB. Halve it.\n"
    "        block_kv = 64\n"
)
if anchor not in src:
    print(f"[{MARKER}] primary anchor not found; trying loose match");
    # loose fallback: replace a standalone 'block_kv = 128' following 'if not use_gluon:'
    loose = "        block_kv = 128\n"
    if "if not use_gluon:" in src and loose in src:
        src = src.replace(loose, "        block_kv = 64  # " + MARKER + " (gfx942 64KB LDS)\n", 1)
        open(F, "w").write(src)
        print(f"[{MARKER}] applied via loose match to {F}")
        sys.exit(0)
    print(f"[{MARKER}] anchor not found — SKIP"); sys.exit(0)

src = src.replace(anchor, replacement, 1)
open(F, "w").write(src)
print(f"[{MARKER}] applied to {F}")
