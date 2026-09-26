#!/usr/bin/env python3
# Fix the torch.compile / Dynamo break blocking decode PIECEWISE cudagraph on GLM-5.3-Flash.
# Root: qwen_gdn_linear_attn.py `_can_use_fused_gdn_mtp_decode` (called from the decode forward
# _forward_core_fused_norm) does `hasattr(torch.ops._C, "fused_gdn_decode_post_conv_mtp")`.
# Dynamo cannot trace hasattr on the torch.ops._C op namespace -> "Unsupported hasattr(TritonKernel...)"
# -> torch.compile aborts -> no cudagraph -> decode runs eager (TPOT ~130ms).
# NOT a missing-kernel issue: the op simply isn't built on gfx942, so the check returns False anyway;
# but the hasattr CALL itself breaks tracing. Fix: compute the boolean ONCE at import (module global),
# then reference the constant inside the traced method (Dynamo-safe).
# Idempotent + anchor-based + self-skip.
import io, sys
F = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"
src = io.open(F, encoding="utf-8").read()
if "GLM53_GDN_HASATTR_FIX" in src:
    print("[patch_gdn_hasattr] already applied"); sys.exit(0)

# 1) inject a module-level constant computed once at import (after the torch import).
#    Place it before the first class, but ABOVE any decorator that precedes that
#    class — inserting between a @decorator and its class is a SyntaxError.
import re
m = re.search(r"^class ", src, re.M)
if not m:
    print("[patch_gdn_hasattr] no class anchor"); sys.exit(2)
insert_at = m.start()
# walk backwards over contiguous decorator lines (and blank lines between them)
lines_before = src[:insert_at].split("\n")
i = len(lines_before) - 1
# lines_before[-1] is the '' just before 'class' (src[:start] ends at line start)
j = i - 1
while j >= 0 and (lines_before[j].lstrip().startswith("@") or lines_before[j].strip() == ""):
    if lines_before[j].lstrip().startswith("@"):
        i = j  # remember the topmost decorator line index
    j -= 1
insert_at = len("\n".join(lines_before[:i]))
if i < len(lines_before) and insert_at > 0:
    insert_at += 1  # keep the newline before the decorator
const_block = (
    "# GLM53_GDN_HASATTR_FIX: compute the fused-GDN-MTP op availability ONCE at import so the traced\n"
    "# decode forward doesn't call hasattr(torch.ops._C, ...) (Dynamo-untraceable -> breaks compile).\n"
    "_HAS_FUSED_GDN_DECODE_POST_CONV_MTP = hasattr(torch.ops._C, \"fused_gdn_decode_post_conv_mtp\")\n\n\n"
)
src = src[:insert_at] + const_block + src[insert_at:]

# 2) replace the in-forward hasattr with the constant.
old = "            and hasattr(torch.ops._C, \"fused_gdn_decode_post_conv_mtp\")"
new = "            and _HAS_FUSED_GDN_DECODE_POST_CONV_MTP"
if old not in src:
    print("[patch_gdn_hasattr] forward-hasattr anchor NOT found"); sys.exit(2)
src = src.replace(old, new, 1)

# 3) also fix the diagnostic one at ~555 (not in forward, but harmless to make consistent).
old2 = "        if not hasattr(torch.ops._C, \"fused_gdn_decode_post_conv_mtp\"):"
new2 = "        if not _HAS_FUSED_GDN_DECODE_POST_CONV_MTP:"
if old2 in src:
    src = src.replace(old2, new2, 1)

io.open(F, "w", encoding="utf-8").write(src)
print("[patch_gdn_hasattr] applied OK")
