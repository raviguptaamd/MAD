# GLM-5.3-Flash disaggregated (gfx942) — patch set

These are the verified vLLM patches that make GLM-5.3-Flash serve correctly on
**AMD MI300X / MI325X (gfx942)** — eager serve + decode CUDA graph + long-context
recall, plus the MoRIIO KV-transfer fix that makes **disaggregated (1P/1D)** recall
correct. Unlike PR254 (gfx950, from-source compile), this recipe is an **OVERLAY**
on the pinned vLLM ROCm nightly: the nightly already ships `glm5next` natively + the
MoRIIO connector + a prebuilt aiter + mori, so our fixes are pure-Python `.py`
overlays + one unified diff, applied idempotently over dist-packages (no
vLLM/aiter/mori recompile). A wholesale PR254-style overlay REGRESSED this stack
(garbage at 24 tokens) because of a connector API skew — see
`moriio_kbpb_fix/README.md`; only the block-routing *idea* was ported.

`apply_patches.sh` applies `core/` (all topologies). `core/14_*.patch` is the recall
fix. `moriio_kbpb_fix/` is disagg-only (KV transfer over MoRIIO). The Dockerfile
bakes all three in order (core → patch14 → kbpb).

## Manifest — file → what it fixes → in-container destination

### core/ — eager serve + decode CUDA graph + recall (COLOCATED + disagg, all need these)

| patch | fixes | destination (dist-packages) |
|---|---|---|
| `01_amd_indexer_dispatch.py` | ROCm dispatch alias for the DSA sparse indexer (`GLM53_AMD_INDEXER_DISPATCH`) | `vllm/models/glm5next/amd/sparse_indexer.py` |
| `02_rocm_topk_ready.py` | ROCm top-k readiness stub (eager serve) | `vllm/…` (anchor-based) |
| `03_tilelang_warmup.py` | tilelang warmup guard (eager serve) | `vllm/…` |
| `04_torch_compile.py` | `@support_torch_compile` gate (eager serve) | `vllm/…` |
| `05_gdn_hasattr.py` | gated-delta-net `hasattr` hoist (eager serve syntax) | `vllm/…` |
| `06_kpool_custom_op.py` | register the kpool indexer as an opaque `torch.ops.vllm` op so Dynamo never traces its `@triton.jit` kernels → PIECEWISE decode cudagraph captures (1st tracing wall) (`GLM53_KPOOL_CUSTOM_OP`) | `vllm/models/glm5next/amd/sparse_indexer.py` |
| `07_fused_qk_rmsnorm_op.py` | same opaque-op treatment for the fused QK-RMSNorm (2nd tracing wall) | `vllm/…` |
| `11_flydsl_shrui_fix.py` | fast prefill MQA-logits (flydsl codegen) | `vllm/…` |
| `14_kpool_slot_mapping.patch` | **THE recall fix** (block-table granularity): expands each coarse shared hybrid-KV block into its `factor` fine indexer pages so every DSA pool gets a unique monotonic slot (was collapsing ~7 pools onto one physical slot → nondeterministic long-context recall). Fixes both the write (`compressed_slot_mapping`) and the read (prefill chunk `block_table`). `patch -p1 --forward`, marker-guarded `GLM53_KPOOL_SLOT_MAPPING_FIX`. | `vllm/v1/attention/backends/mla/indexer.py` |

> Provisional patches 09/10/12 are NOT applied (09 superseded by 11; 10/12 proven
> irrelevant to recall). `apply_patches.sh` runs exactly `01-07,11`.

### moriio_kbpb_fix/ — disagg-only KV-transfer recall fix (MoRIIO 1P/1D)

| file | role | destination |
|---|---|---|
| `moriio_layout.py` | **carries the fix**: `compute_block_transfer_offsets` expands each shared group block-id into its `kbpb` kernel sub-blocks before byte offsets (`GLM53_INDEXER_KBPB`). `kbpb = per_layer_num_blocks / reference_group_block_count`, derived per leg from the MoRIIO handshake metadata. Fires only on `indexer.k_cache` (kbpb=17 on EP8, self-adapts to 9 on TP4); every other layer is kbpb=1 → byte-identical no-op. | `vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_layout.py` |
| `moriio_connector.py` | **carries the fix**: passes the two reference group-block counts (`local_num_blocks`, `remote_ref_blocks`) into the layout expansion. | `.../moriio/moriio_connector.py` |
| `moriio_common.py` | unchanged native (included for a complete matched set) | `.../moriio/moriio_common.py` |
| `moriio_engine.py` | unchanged native (included for a complete matched set) | `.../moriio/moriio_engine.py` |
| `niah_disagg.py`, `perf_disagg.py`, `perf_sweep_async.py` | test/bench harnesses (NIAH recall + TTFT/TPOT sweep over the disagg proxy) | — (harness, not overlaid) |

See `moriio_kbpb_fix/README.md` for the full root-cause writeup (the ~4096-token
recall wall, `MORIIO_OFFSET_DBG=1` instrumentation, before/after grid).

## Fidelity

These are the files verified on-device (2026-09-23/24) on gfx942 (MI300X 192GB /
MI325X 256GB): EP8/EP8 NIAH 21/21 all depths to 256K, TP4/TP4 clean to ~62K. The
`serve/tp4/bundle/` copy of `core/` + `moriio_kbpb_fix/` is byte-identical to this
`patches/` set (the Dockerfile bakes this one). Do not edit in place — regenerate +
re-verify if a fix changes.
