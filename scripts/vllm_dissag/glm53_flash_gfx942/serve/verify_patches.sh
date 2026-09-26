#!/bin/bash
D=/usr/local/lib/python3.12/dist-packages/vllm
echo "dispatch:  $(grep -c GLM53_AMD_INDEXER_DISPATCH $D/models/glm5next/amd/sparse_indexer.py 2>/dev/null)"
echo "topk:      $(grep -c GLM53_ROCM_TOPK_READY $D/v1/attention/backends/mla/rocm_aiter_mla_sparse.py 2>/dev/null)"
echo "tilelang:  $(grep -c GLM53_TILELANG_WARMUP_GUARD $D/model_executor/warmup/jit_warmup_tilelang_helper.py 2>/dev/null)"
echo "decorator: $(grep -c GLM53_TORCH_COMPILE $D/models/glm5next/common/model.py 2>/dev/null)"
echo "gdn:       $(grep -c GLM53_GDN_HASATTR_FIX $D/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py 2>/dev/null)"
echo "eager serving now?: $(curl -s http://localhost:20066/health -o /dev/null -w '%{http_code}' 2>/dev/null)"
