#!/usr/bin/env python3
# Fix: mla.py:235 calls self.mla_attn.impl.record_logical_topk_ready() but the nightly's
# ROCMAiterMLASparseImpl predates the "index_group" refactor and doesn't define it ->
# AttributeError -> EngineCore init crash during warmup. On the ROCm single-group path index_group
# is None, so the method is a no-op (the base impl only acts when self.index_group is not None).
# Add a no-op record_logical_topk_ready (+ prepare_for_batch guard) to ROCMAiterMLASparseImpl.
# Idempotent + anchor-based + self-skip.
import io, sys
F = "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py"
src = io.open(F, encoding="utf-8").read()
if "GLM53_ROCM_TOPK_READY" in src:
    print("[patch_rocm_topk_ready] already applied"); sys.exit(0)

# insert the stubs right after the class attribute block (after 'supports_dcp = False').
anchor = "    is_sparse = True\n    supports_dense_mha_prefill = False\n    supports_dcp = False\n"
if anchor not in src:
    print("[patch_rocm_topk_ready] ANCHOR NOT FOUND"); sys.exit(2)

inject = anchor + (
    "\n"
    "    # GLM53_ROCM_TOPK_READY: the newer sparse-MLA interface (mla.py) calls\n"
    "    # record_logical_topk_ready()/prepare_for_batch() for the index_group DCP path.\n"
    "    # The ROCm impl is single-group (no index_group), so these are no-ops.\n"
    "    def record_logical_topk_ready(self) -> None:\n"
    "        return None\n"
    "\n"
    "    def prepare_for_batch(self, attn_metadata=None) -> None:\n"
    "        return None\n"
)
src = src.replace(anchor, inject, 1)
io.open(F, "w", encoding="utf-8").write(src)
print("[patch_rocm_topk_ready] applied OK")
