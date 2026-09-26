#!/usr/bin/env python3
# Overlay fix for p9 image bug: kernel_warmup -> compile_tilelang() calls
# jit_impl.compile(*args) but for some DSA kernels jit_impl is a bare `function`
# (no `.compile`), raising: AttributeError: 'function' object has no attribute 'compile'
# → EngineCore init fails on prefill/TP8.
#
# Fix: guard compile_tilelang to no-op when jit_impl lacks a callable `.compile`.
# The kernel still JIT-compiles lazily on first real use — only the pre-warm is skipped.
# Idempotent + anchor-based (degrades to no-op if the file was already patched/refactored).
import io, re, sys

F = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/warmup/jit_warmup_tilelang_helper.py"

src = io.open(F, encoding="utf-8").read()
if "GLM53_TILELANG_WARMUP_GUARD" in src:
    print("[patch_tilelang_warmup] already applied"); sys.exit(0)

anchor = "    with _quiet_tilelang_warmup_logs():\n        compiled = jit_impl.compile(*args, **kwargs)"
if anchor not in src:
    print("[patch_tilelang_warmup] ANCHOR NOT FOUND — aborting (no change)"); sys.exit(2)

replacement = (
    "    # GLM53_TILELANG_WARMUP_GUARD: some DSA kernels pass a bare function as jit_impl\n"
    "    # (no .compile) — skip pre-warm for those; they JIT lazily at first use.\n"
    "    _compile = getattr(jit_impl, \"compile\", None)\n"
    "    if not callable(_compile):\n"
    "        return\n"
    "    with _quiet_tilelang_warmup_logs():\n"
    "        compiled = _compile(*args, **kwargs)"
)
src = src.replace(anchor, replacement, 1)
io.open(F, "w", encoding="utf-8").write(src)
print("[patch_tilelang_warmup] applied OK")
