#!/usr/bin/env python3
# L2c — register fused_q_kv_rmsnorm as an opaque custom op (same proven primitive
# as L2 kpool). After L2, the PIECEWISE trace advanced past the indexer and hit
# the SAME raw-triton `arg_names` break in fused_q_kv_rmsnorm (mla.py:208 ->
# fused_qk_rmsnorm.py:211 _FUSED_Q_KV_RMSNORM_KERNEL). torch.compiler.disable was
# WRONG (graph-break, fatal under vLLM fullgraph=True PIECEWISE, gb0099). The
# reference primitive is direct_register_custom_op -> opaque FX leaf.
#
# fused_q_kv_rmsnorm(qr, kv, q_weight, kv_weight, eps) -> (qr_out, kv_out):
# all-tensor + float in, two FRESH tensors out, no input mutation -> mutates_args=().
#
# PATCH (file-level so mla.py's `from ... import fused_q_kv_rmsnorm` picks up the
# op-wrapped name at import): append a block that captures the original fn,
# registers it as torch.ops.vllm.glm53_fused_qk_rmsnorm (+fake), and rebinds the
# module-level `fused_q_kv_rmsnorm` to call the op.
#
# !!! UNTESTED on hardware — validate with the live PIECEWISE capture.
# Marker: GLM53_FUSED_QK_RMSNORM_OP  Target: vllm/models/common/ops/fused_qk_rmsnorm.py
import os
import sys

MARKER = "GLM53_FUSED_QK_RMSNORM_OP"
CANDIDATES = [
    "/usr/local/lib/python3.12/dist-packages/vllm/models/common/ops/fused_qk_rmsnorm.py",
]
if len(sys.argv) > 1:
    CANDIDATES.insert(0, sys.argv[1])

target = next((p for p in CANDIDATES if os.path.isfile(p)), None)
if target is None:
    print(f"[{MARKER}] target fused_qk_rmsnorm.py not found; SKIP")
    sys.exit(0)

src = open(target, "r").read()
if MARKER in src:
    print(f"[{MARKER}] already applied in {target}; SKIP")
    sys.exit(0)

if "def fused_q_kv_rmsnorm(" not in src:
    print(f"[{MARKER}] anchor `def fused_q_kv_rmsnorm(` missing; build differs — SKIP")
    sys.exit(0)

BLOCK = f'''

# ===================== {MARKER} (L2c: opaque custom-op) =====================
# Make fused_q_kv_rmsnorm an opaque torch.ops.vllm.* op so PIECEWISE cudagraph
# capture does not trace into its raw @triton.jit kernel.
import torch as _t_glm53
from vllm.logger import init_logger as _init_logger_glm53
_logger_glm53 = _init_logger_glm53(__name__)

try:
    try:
        from vllm.utils.torch_utils import direct_register_custom_op as _drc_glm53
    except Exception:
        from vllm.utils import direct_register_custom_op as _drc_glm53

    _glm53_orig_fused_q_kv_rmsnorm = fused_q_kv_rmsnorm

    def _glm53_fused_qk_rmsnorm_op(
        qr: _t_glm53.Tensor,
        kv: _t_glm53.Tensor,
        q_weight: _t_glm53.Tensor,
        kv_weight: _t_glm53.Tensor,
        eps: float,
    ) -> tuple[_t_glm53.Tensor, _t_glm53.Tensor]:
        return _glm53_orig_fused_q_kv_rmsnorm(qr, kv, q_weight, kv_weight, eps)

    def _glm53_fused_qk_rmsnorm_fake(
        qr, kv, q_weight, kv_weight, eps,
    ):
        return _t_glm53.empty_like(qr), _t_glm53.empty_like(kv)

    _drc_glm53(
        op_name="glm53_fused_qk_rmsnorm",
        op_func=_glm53_fused_qk_rmsnorm_op,
        mutates_args=[],
        fake_impl=_glm53_fused_qk_rmsnorm_fake,
    )

    def fused_q_kv_rmsnorm(qr, kv, q_weight, kv_weight, eps):  # noqa: F811
        return _t_glm53.ops.vllm.glm53_fused_qk_rmsnorm(qr, kv, q_weight, kv_weight, eps)

    _logger_glm53.info("{MARKER}: fused_q_kv_rmsnorm registered as opaque custom op "
                       "torch.ops.vllm.glm53_fused_qk_rmsnorm")
except Exception as _e_glm53:
    _logger_glm53.warning("{MARKER}: registration FAILED (%s); keeping raw fn "
                          "(eager ok, PIECEWISE still broken)", _e_glm53)
# =================== end {MARKER} ===================
'''

with open(target, "a") as f:
    f.write(BLOCK)
print(f"[{MARKER}] applied to {target} (UNTESTED — validate with live PIECEWISE capture)")
