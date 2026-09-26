#!/usr/bin/env python3
# Fix: on gfx942/ROCm, the AMD SparseAttnIndexerKpool CustomOp only implements forward_native (which
# calls torch.ops.vllm.rocm_aiter_sparse_attn_indexer). But when the op is "enabled" in dispatch_forward,
# ROCm routes forward_hip -> forward_cuda -> raise NotImplementedError (base defaults), NEVER reaching
# forward_native. Result: EngineCore init crashes with NotImplementedError during the DSA indexer forward.
# Fix: alias forward_cuda (and thus the inherited forward_hip) to the AMD forward_native so the ROCm
# dispatch reaches the real aiter implementation regardless of enabled/disabled state.
# Idempotent + anchor-based + self-skip.
import io, sys
F = "/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/amd/sparse_indexer.py"
src = io.open(F, encoding="utf-8").read()
if "GLM53_AMD_INDEXER_DISPATCH" in src:
    print("[patch_amd_indexer_dispatch] already applied"); sys.exit(0)

# insert forward_cuda = forward_native alias right after the class's forward_native def signature block.
# Anchor on the def forward_native inside SparseAttnIndexerKpool.
anchor = "    def forward_native(\n        self,\n        hidden_states: torch.Tensor,\n        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],\n        k: torch.Tensor,\n        weights: torch.Tensor,"
if anchor not in src:
    print("[patch_amd_indexer_dispatch] ANCHOR NOT FOUND"); sys.exit(2)

inject = ("    # GLM53_AMD_INDEXER_DISPATCH: route ROCm forward_cuda/forward_hip to the AMD native impl\n"
          "    # (which calls the aiter sparse_attn_indexer op). Base CustomOp forward_cuda raises\n"
          "    # NotImplementedError, and forward_hip defaults to forward_cuda -> crash on gfx942.\n"
          "    def forward_cuda(self, *args, **kwargs):\n"
          "        return self.forward_native(*args, **kwargs)\n\n"
          + anchor)
src = src.replace(anchor, inject, 1)
io.open(F, "w", encoding="utf-8").write(src)
print("[patch_amd_indexer_dispatch] applied OK")
