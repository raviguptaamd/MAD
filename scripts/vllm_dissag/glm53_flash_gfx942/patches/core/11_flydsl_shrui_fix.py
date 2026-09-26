#!/usr/bin/env python3
# PERF (task #65) — fix the flydsl fp8_mqa_logits codegen crash at its ROOT so the
# FAST flydsl path can be re-enabled (instead of the slower triton/torch bypass 09).
#
# BUG: aiter/ops/flydsl/kernels/mqa_logits/fp8_mqa_logits.py _fn_to_fnuz_i64:
#   line 203  hi_i32 = fx.Int32(raw.shrui(32))
#   line 209  byte_val = src.shrui(shift) & 0xFF        # shift = byte_idx*8 (python int)
# flydsl ShRUIOp calls _to_raw(amount); a bare python int has no .ir_value()/_CAPIPtr
# -> AttributeError 'int' object has no attribute '_CAPIPtr' -> whole kernel fails to
# compile -> server dies on any >2K prefill.
#
# FIX: wrap the shift AMOUNTS as flydsl Int32 constants so _to_raw resolves them:
#   raw.shrui(32)      -> raw.shrui(fx.Int64(32))
#   src.shrui(shift)   -> src.shrui(fx.Int32(shift))
# Minimal, behavior-preserving (same shift semantics), lets flydsl emit the op.
#
# After this applies cleanly AND is validated (no crash + correct logits + perf gain),
# revert patch 09 (flydsl bypass) so the fast path is used.
#
# Marker: GLM53_FLYDSL_SHRUI_FIX
# Target: aiter/ops/flydsl/kernels/mqa_logits/fp8_mqa_logits.py
import os, sys

MARKER = "GLM53_FLYDSL_SHRUI_FIX"
CANDIDATES = [
    "/usr/local/lib/python3.12/dist-packages/aiter/ops/flydsl/kernels/mqa_logits/fp8_mqa_logits.py",
]
if len(sys.argv) > 1:
    CANDIDATES.insert(0, sys.argv[1])
F = next((p for p in CANDIDATES if os.path.isfile(p)), None)
if F is None:
    print(f"[{MARKER}] target not found; SKIP"); sys.exit(0)

src = open(F).read()
if MARKER in src:
    print(f"[{MARKER}] already applied; SKIP"); sys.exit(0)

changed = False
# hunk 1: raw.shrui(32)
a1 = "            hi_i32 = fx.Int32(raw.shrui(32))"
r1 = ("            # " + MARKER + ": wrap shift amount as fx.Int32 (flydsl _to_raw needs an ir value)\n"
      "            hi_i32 = fx.Int32(raw.shrui(fx.Int64(32)))")
if a1 in src:
    src = src.replace(a1, r1, 1); changed = True; print(f"[{MARKER}] hunk1 (shrui 32) applied")
else:
    print(f"[{MARKER}] hunk1 anchor not found")

# hunk 2: src.shrui(shift)
a2 = "                    byte_val = src.shrui(shift) & 0xFF"
r2 = "                    byte_val = src.shrui(fx.Int32(shift)) & 0xFF  # " + MARKER
if a2 in src:
    src = src.replace(a2, r2, 1); changed = True; print(f"[{MARKER}] hunk2 (shrui shift) applied")
else:
    print(f"[{MARKER}] hunk2 anchor not found")

if changed:
    open(F, "w").write(src)
    # clear stale flydsl JIT cache so the kernel recompiles with the fix
    for c in ("/root/.aiter", "/opt/vllm_cache/aiter_jit"):
        try:
            import shutil; shutil.rmtree(c, ignore_errors=True)
        except Exception:
            pass
    print(f"[{MARKER}] written to {F} (+cleared aiter JIT cache)")
else:
    print(f"[{MARKER}] no change")
