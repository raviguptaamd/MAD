#!/usr/bin/env python3
# L2 — THE CRUX. Register the GLM-5.3 kpool sparse-attn indexer as an opaque
# torch.ops.vllm.* custom op so Dynamo never traces into its raw @triton.jit
# kernels -> PIECEWISE decode cudagraph captures (no hasattr(TritonKernel,arg_names)).
#
# WHY: GLM-5.3 asserts index_kpool>1 (common/attention.py:114) -> forward_hip ->
# forward_cuda -> the BARE fn sparse_attn_indexer_kpool (amd/sparse_indexer.py:259),
# decorated ONLY @eager_break_during_capture and NEVER direct_register_custom_op'd.
# DSV4 + K3 keep their indexer ALWAYS behind direct_register_custom_op
# (_aiter_ops.py:2439) -> opaque -> standard PIECEWISE captures fine. We give the
# kpool op the identical boundary.
#
# APPROACH (idempotent, appended, self-skip): append a marked block at end of the
# module that (1) defines a clean-typed wrapper calling the existing bare fn,
# (2) registers it via direct_register_custom_op with a fake + mutates_args, and
# (3) rebinds SparseAttnIndexerKpool.forward_cuda to route through the registered
# op. We do NOT surgically rewrite forward_cuda in place (fragile); rebinding the
# method post-registration is equivalent and robust.
#
# mutates_args = the 3 tensors the op writes in place:
#   kv_cache (k-cache insert), topk_indices_buffer (result), tail_kv_cache (tail insert).
#
# !!! UNTESTED on hardware — must be validated with a live PIECEWISE capture on a
#     gfx942 node (008 build + compute node). See docs/LAYERED_PLAN.md open Qs.
#
# Marker: GLM53_KPOOL_CUSTOM_OP   Target: vllm/models/glm5next/amd/sparse_indexer.py
import os
import sys

MARKER = "GLM53_KPOOL_CUSTOM_OP"
CANDIDATES = [
    "/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/amd/sparse_indexer.py",
]
# allow override: python3 06_kpool_custom_op.py /path/to/sparse_indexer.py
if len(sys.argv) > 1:
    CANDIDATES.insert(0, sys.argv[1])

target = next((p for p in CANDIDATES if os.path.isfile(p)), None)
if target is None:
    print(f"[{MARKER}] target sparse_indexer.py not found in {CANDIDATES}; SKIP")
    sys.exit(0)

src = open(target, "r").read()
if MARKER in src:
    print(f"[{MARKER}] already applied in {target}; SKIP")
    sys.exit(0)

# Sanity: the anchors we depend on must exist in this build.
need = [
    "def sparse_attn_indexer_kpool(",
    'class SparseAttnIndexerKpool(CustomOp)',
    "def forward_cuda(",
]
missing = [n for n in need if n not in src]
if missing:
    print(f"[{MARKER}] EXPECTED ANCHORS MISSING {missing} in {target}; "
          f"build may differ from pinned f169e8df — SKIP (do not force)")
    sys.exit(0)

BLOCK = f'''

# ===================== {MARKER} (L2: opaque custom-op) =====================
# Registers the kpool indexer as torch.ops.vllm.glm53_sparse_attn_indexer_kpool so
# Dynamo treats it as an opaque leaf (its raw @triton.jit kernels are hidden),
# enabling PIECEWISE decode cudagraph. Rebinds forward_cuda to route through it.
import torch as _t_glm53
from vllm.logger import init_logger as _init_logger_glm53
_logger_glm53 = _init_logger_glm53(__name__)

try:
    try:
        from vllm.utils.torch_utils import direct_register_custom_op as _drc_glm53
    except Exception:  # older layout
        from vllm.utils import direct_register_custom_op as _drc_glm53

    def _glm53_kpool_indexer_op(
        hidden_states: _t_glm53.Tensor,
        k_cache_prefix: str,
        kv_cache: _t_glm53.Tensor,
        q_quant: _t_glm53.Tensor,
        q_scale: _t_glm53.Tensor | None,
        k: _t_glm53.Tensor,
        weights: _t_glm53.Tensor,
        quant_block_size: int,
        scale_fmt: str | None,
        topk_tokens: int,
        head_dim: int,
        max_model_len: int,
        total_seq_lens: int,
        topk_indices_buffer: _t_glm53.Tensor,
        skip_k_cache_insert: bool,
        use_fp4_cache: bool,
        gate_score: _t_glm53.Tensor | None,
        compress_ape: _t_glm53.Tensor | None,
        index_kpool: int,
        positions: _t_glm53.Tensor | None,
        tail_kv_cache: _t_glm53.Tensor | None,
        tail_prefix: str | None,
    ) -> _t_glm53.Tensor:
        # Calls the existing bare fn (its @eager_break_during_capture is a no-op
        # passthrough when breakable is off; and we are inside an opaque op anyway).
        return sparse_attn_indexer_kpool(
            hidden_states, k_cache_prefix, kv_cache, q_quant, q_scale, k, weights,
            quant_block_size, scale_fmt, topk_tokens, head_dim, max_model_len,
            total_seq_lens, topk_indices_buffer, skip_k_cache_insert, use_fp4_cache,
            gate_score, compress_ape, index_kpool, positions, tail_kv_cache, tail_prefix,
        )

    def _glm53_kpool_indexer_op_fake(
        hidden_states, k_cache_prefix, kv_cache, q_quant, q_scale, k, weights,
        quant_block_size, scale_fmt, topk_tokens, head_dim, max_model_len,
        total_seq_lens, topk_indices_buffer, skip_k_cache_insert, use_fp4_cache,
        gate_score, compress_ape, index_kpool, positions, tail_kv_cache, tail_prefix,
    ):
        # The op writes topk_indices_buffer in place and returns it. The fake MUST
        # return the actual input buffer (identity-preserving), NOT empty_like:
        # under cudagraph replay a fresh tensor changes address each replay and the
        # downstream read sees stale data. This matches the reference op
        # rocm_aiter_sparse_attn_indexer_fake (rocm_aiter_mla_sparse.py:977 -> return
        # topk_indices_buffer).
        return topk_indices_buffer

    _drc_glm53(
        op_name="glm53_sparse_attn_indexer_kpool",
        op_func=_glm53_kpool_indexer_op,
        # Match reference: only topk_indices_buffer is a declared mutation. kv_cache /
        # tail are persistent static out-of-graph side buffers; declaring them makes
        # functionalization clone them and breaks static-address assumptions.
        mutates_args=["topk_indices_buffer"],
        fake_impl=_glm53_kpool_indexer_op_fake,
    )

    def _glm53_forward_cuda(
        self,
        hidden_states,
        q_quant,
        k,
        weights,
        *,
        gate_score=None,
        compress_ape=None,
        index_kpool: int = 1,
        positions=None,
    ):
        if isinstance(q_quant, tuple):
            q_values, q_scale = q_quant
        else:
            q_values, q_scale = q_quant, None
        return _t_glm53.ops.vllm.glm53_sparse_attn_indexer_kpool(
            hidden_states,
            self.k_cache.prefix,
            self.k_cache.kv_cache,
            q_values,
            q_scale,
            k,
            weights,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
            self.skip_k_cache_insert,
            self.use_fp4_cache,
            gate_score,
            compress_ape,
            index_kpool,
            positions,
            self.tail_cache.kv_cache if self.tail_cache is not None else None,
            self.tail_cache.prefix if self.tail_cache is not None else None,
        )

    SparseAttnIndexerKpool.forward_cuda = _glm53_forward_cuda
    # CRUCIAL: under torch.compile, CustomOp.dispatch_forward takes the compile_native
    # path (custom_op.py:190-192) -> forward_NATIVE, not forward_cuda/forward_hip.
    # Bind forward_native too, else the op is never on the traced path and Dynamo
    # traces the bare fn (garbage / hasattr break). _glm53_forward_cuda is signature-
    # compatible with forward_native (single-tensor q_quant -> q_scale=None).
    SparseAttnIndexerKpool.forward_native = _glm53_forward_cuda
    _logger_glm53.info("{MARKER}: kpool indexer registered as opaque custom op; "
                       "forward_native+forward_cuda routed through torch.ops.vllm.glm53_sparse_attn_indexer_kpool")
except Exception as _e_glm53:  # never brick eager serving on a registration failure
    _logger_glm53.warning("{MARKER}: registration FAILED (%s); "
                          "falling back to bare fn (eager ok, PIECEWISE still broken)", _e_glm53)
# =================== end {MARKER} ===================
'''

with open(target, "a") as f:
    f.write(BLOCK)
print(f"[{MARKER}] applied to {target} (UNTESTED — validate with live PIECEWISE capture)")
