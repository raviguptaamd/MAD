#!/usr/bin/env python3
# Enable torch.compile (required for PIECEWISE/decode cudagraph) on GLM-5.3-Flash by adding the
# @support_torch_compile decorator to Glm5NextModel — the model class is missing it, so vLLM logs
# "torch.compile is turned on, but the model does not support it" and decode falls back to eager
# (TPOT ~130ms). Mirrors the afmoe.py pattern. dynamic_arg_dims marks the token dim per arg.
# Idempotent + anchor-based + self-skip.
import io, sys
F = "/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/common/model.py"
src = io.open(F, encoding="utf-8").read()
if "GLM53_TORCH_COMPILE" in src:
    print("[patch_torch_compile] already applied"); sys.exit(0)

# 1) add the import (after an existing vllm.compilation or vllm.config import)
import_anchor = "from vllm.config import"
if "from vllm.compilation.decorators import support_torch_compile" not in src:
    if import_anchor in src:
        src = src.replace(import_anchor,
            "from vllm.compilation.decorators import support_torch_compile  # GLM53_TORCH_COMPILE\n"
            + import_anchor, 1)
    else:
        print("[patch_torch_compile] import anchor not found"); sys.exit(2)

# 2) decorate Glm5NextModel
class_anchor = "class Glm5NextModel(nn.Module):"
if class_anchor not in src:
    print("[patch_torch_compile] class anchor not found"); sys.exit(2)
decorator = (
    "@support_torch_compile(\n"
    "    dynamic_arg_dims={\n"
    "        \"input_ids\": 0,\n"
    "        \"positions\": -1,\n"
    "        \"intermediate_tensors\": 0,\n"
    "        \"inputs_embeds\": 0,\n"
    "    }\n"
    ")\n"
    + class_anchor
)
src = src.replace(class_anchor, decorator, 1)
io.open(F, "w", encoding="utf-8").write(src)
print("[patch_torch_compile] applied OK")
