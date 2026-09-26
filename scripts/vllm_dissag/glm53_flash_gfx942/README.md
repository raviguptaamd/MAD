# GLM-5.3-Flash-FP8 disaggregated (1P/1D) on AMD Instinct gfx942 (MI300X / MI325X) + MoRIIO

`GLM-5.3-Flash` (`Glm5NextForConditionalGeneration`), **FP8** (E4M3, dynamic activation
scaling), is a DeepSeek-DSA / MLA + KDA-linear-attention hybrid with an MTP head. This
recipe serves it **disaggregated** (prefill / decode split across nodes) over the
**MoRIIO** RDMA KV-transfer connector on **AMD MI300X (192 GB) / MI325X (256 GB) — gfx942**
(same recipe both SKUs).

This is the **gfx942 sibling** of ROCm/MAD PR254 (gfx950 / MI355X): same structure and
discipline, built **from source** on the ROCm dev base.

> **STATUS: validation in progress.** This recipe is under active bring-up. Recall / perf
> numbers are intentionally NOT published here yet; they will be added once measured and
> confirmed on this stack. Nothing below should be read as a validated performance claim.

## Topologies (one image, launch-env selects the topology)
- **EP8 1P/1D** — DP8 + expert-parallel per leg, MoRIIO KV transfer + allgather/reducescatter
  MoE dispatch. Lead config.
- **TP4×DP2 1P/1D** — TP4 × DP2 per leg.
- **TP4/TP4 1P/1D** — TP4 per leg (DP1).

## The stack (from source — latest)
Built from `Dockerfile` (from-source, no prebuilt-vLLM overlay):

| Subcomponent | Pin |
|---|---|
| **base image** | `rocm/vllm-dev:ci_base-build-01a0d6a8-…` (ROCm 7.2.3 dev/build base, no bundled vLLM) |
| **vLLM** | fork `@41644d2` + cherry-picks `0344b77e2 cda364860 e8c186f71 623fdc946` (glm5next-native) |
| **MoRI** | source `@78b7a5f3` (gfx942, NIC backends) |
| **aiter** | fork `@b50066a9` + flydsl `0.2.2` |
| **vllm-router** | upstream `vllm-project/router @0fb97775` + dpfix `raviguptaamd/router @82dc9811` (Rust, from source) |
| arch target | `gfx942` (MI300X / MI325X) |

GLM-5.3 fixes are applied as the `patches/` set on top of the from-source vLLM tree
(see `patches/` + the Dockerfile). Exact build args and pins are in the Dockerfile header.

## The DSA disagg fixes (why this recipe exists)
Serving GLM-5.3-Flash disaggregated with correct long-context recall on gfx942 requires,
beyond the model integration patches (`patches/core/`):

1. **kpool slot-mapping** (`patches/core/14_*.patch`) — expands each coarse shared hybrid-KV
   block into its fine DSA-indexer pages so every sparse-indexer pool gets a unique monotonic
   slot (colocated recall correctness).
2. **MoRIIO per-group block expansion** (`patches/moriio_kbpb_fix/`) — the DSA `indexer.k_cache`
   is paged at a finer kernel block (`kbpb`) than the shared group block, so each group block-id
   must be expanded into its `kbpb` kernel sub-blocks before byte offsets are computed, or the
   KV transfer reads the wrong offset → recall corruption past the sparsity window.
3. **DSA-prefill kernel fallback** (`patches/core/18`,`19`) — provides a triton fallback +
   gfx942-LDS tiling for the fp8 MQA-logits path.
4. **DSA-indexer PD read barrier** (`patches/core/21`) — the sparse indexer waits for its
   MoRIIO KV read to land before use, mirroring the dense-layer barrier.

## Mandatory config-level fixes (serve flags/env — in the serve scripts)
| item | value | why |
|---|---|---|
| `VLLM_USE_BREAKABLE_CUDAGRAPH=1` | env | decode cudagraph correctness (indexer runs outside the graph) |
| `--compilation-config '{"cudagraph_mode":"PIECEWISE"}'` | decode | decode CUDA graph (piecewise; indexer eager) |
| `--max-num-batched-tokens 16384` | both legs | caps per-pass tokens below the int32 workspace-offset kernel ceiling |
| `--no-enable-prefix-caching` | both legs | prefix-cache partial re-prefill faults at long context |
| `--block-size 4` | both legs | must be a multiple of `index_kpool=4` (block-size 1 rejected on GLM-5.3) |
| aiter env | `VLLM_ROCM_USE_AITER=1` (+`_MLA/_MOE/_RMSNORM=1`, `_FP8BMM=0`), `VLLM_USE_DEEP_GEMM=0`, `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=4096` | kernel selection |
| MoRIIO transport | `MORI_IO_CP_TIMEOUT_MS=60000` | control-plane read timeout for in-worker CreateSession under load |
| MoRIIO fabric | `MORI_IB_GID_INDEX=3`, `RDMA_DEVICES=mlx5_*` (MI300) / `rdma0..7` (MI325) | RDMA transport env |

Prefill runs `--enforce-eager`; decode runs PIECEWISE cudagraph.

## Bring-up ORDER (order-sensitive)
1. **Decode leg first** (to "Application startup complete"), **then prefill**. Prefill caches
   the decode mori handshake; restarting decode after prefill is up requires restarting prefill.
2. **Router last** (`serve/router/serve_router.sh`). `RDP` (`--intra-node-data-parallel-size`)
   MUST match the legs' `--data-parallel-size` (8 EP8 / 1 TP4 / 2 TP4×DP2) or requests 400.
3. **Warm** with 2–3 tiny requests (mori CreateSession is a cold-start race; first request may 503).

## Run it
`serve/` holds per-node, per-role launchers (they carry the config fixes above; edit `proxy_ip`
/ model path for your nodes). One long-lived container per node, then exec the role script:

```bash
# EP8/EP8 — decode leg first, then prefill, then router
docker exec -d nite-decode  bash /opt/serve/serve_disagg_ep8_decode.sh
docker exec -d nite-prefill bash /opt/serve/serve_disagg_ep8_prefill.sh
docker exec -d nite-prefill env TOPO=ep8 bash /opt/serve/router/serve_router.sh   # RDP=8
```

Both legs: `--data-parallel-size 8 --enable-expert-parallel --all2all-backend
allgather_reducescatter --max-model-len 270000 --max-num-batched-tokens 16384
--no-enable-prefix-caching --block-size 4`, prefill `--enforce-eager`, decode PIECEWISE.

## Verify
`patches/moriio_kbpb_fix/niah_disagg.py <ntok> <depth> <port> 256` for needle-in-haystack recall.
See `TEST_PLAN.md` for the validation matrix (results tracked separately during bring-up).

## Model registration
Registered in `scripts/vllm_dissag/models.json` + `models.yaml` (see `REGISTER.md`) alongside the
other vllm_disagg models.
