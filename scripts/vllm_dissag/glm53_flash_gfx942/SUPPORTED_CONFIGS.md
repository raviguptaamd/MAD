# GLM-5.3-Flash-FP8 (gfx942) — Supported Configurations & How to Run

`Glm5NextForConditionalGeneration` (`glm5_next`): FP8, MLA + DeepSeek Sparse Attention (DSA,
index_kpool=4) + KDA linear attention + MTP head, 288 routed experts, 1M ctx. Runs on
AMD Instinct **MI300X (192GB) and MI325X (256GB)** — both gfx942, identical recipe.

> **STATUS: validation in progress.** This recipe is under active bring-up on the from-source
> (modern) stack; recall/perf numbers are intentionally NOT published here yet. Results are being
> tracked privately during debug and will be added once measured and confirmed on the shipping stack.
> Status legend (for when populated): ✅ validated · ⚠️ partial · 🚧 experimental.

## Configurations

| # | Config | GPUs | Prefill | Decode | Recall | Status |
|---|--------|------|---------|--------|--------|--------|
| 1 | **EP8/EP8** disagg (1P/1D) | 2×8 | EAGER, DP8+EP | PIECEWISE cudagraph, DP8+EP | _pending_ | 🚧 validating |
| 2 | **TP4×DP2** disagg (1P/1D) | 2×8 | EAGER, TP4×DP2 | PIECEWISE cudagraph, TP4×DP2 | _pending_ | 🚧 validating |
| 3 | **TP4/TP4** disagg (1P/1D) | 2×4 | EAGER, TP4 | PIECEWISE cudagraph, TP4 | _pending_ | 🚧 validating |
| 4 | **Colocated** (single node) | 4 | — | PIECEWISE + MTP | _pending_ | 🚧 validating |

Common to all disagg configs: `--max-model-len 270000 --max-num-batched-tokens 16384
--no-enable-prefix-caching --block-size 4` (block-size MUST be a multiple of index_kpool=4;
block-size 1 is **rejected at load** on GLM-5.3, unlike GLM-5.1). Prefill `--enforce-eager`;
decode `--compilation-config '{"cudagraph_mode":"PIECEWISE"}'` + `VLLM_USE_BREAKABLE_CUDAGRAPH=1`.
Parsers: `--tool-call-parser glm47 --reasoning-parser glm45 --enable-auto-tool-choice`.
KV transfer over MoRIIO; routing via the production `vllm-router` (2P2D KV-notify dpfix).

## How to run (per topology)

Build the image (see docker/vllm_disagg_inference.glmv53flash.mi300.ubuntu.amd.Dockerfile),
launch a long-lived container per node with the model mounted at /models/GLM-5.3-Flash-FP8,
then:

```bash
# router RDP must match the legs' data-parallel-size (8 EP8 / 1 TP4 / 2 TP4xDP2)
# --- EP8/EP8 ---
DECODE:  bash serve/serve_disagg_ep8_decode.sh          # PIECEWISE, DP8+EP
PREFILL: bash serve/serve_disagg_ep8_prefill.sh         # eager, DP8+EP
ROUTER:  TOPO=ep8     bash serve/router/serve_router.sh # RDP=8

# --- TP4xDP2 (both legs DP=2) ---
DECODE:  DP=2 bash serve/tp4/serve_disagg_tp4_decode_nomtp.sh
PREFILL: DP=2 bash serve/tp4/serve_disagg_tp4_prefill.sh
ROUTER:  TOPO=tp4xdp2 MORIIO_DP_SIZE=2 bash serve/router/serve_router.sh  # RDP=2

# --- TP4/TP4 (DP=1) ---
DECODE:  DP=1 bash serve/tp4/serve_disagg_tp4_decode_nomtp.sh
PREFILL: DP=1 bash serve/tp4/serve_disagg_tp4_prefill.sh
ROUTER:  TOPO=tp4     bash serve/router/serve_router.sh  # RDP=1
```
Bring order: decode → prefill → router. Warm 2–3 tiny requests (mori CreateSession cold-start
can 503 the first). Verify with `patches/moriio_kbpb_fix/niah_disagg.py <ntok> <depth> <port> 256`.

## MI325X fabric note
MI325 uses IB HCAs `rdma0..rdma7` (not `mlx5_*`) and socket iface `eno0` (not `eth0`).
Set `RDMA_DEVICES=rdma0,..,rdma7`, `MORI_SOCKET_IFNAME=eno0`, `NCCL_SOCKET_IFNAME=eno0`,
`MORI_IB_GID_INDEX=3`, and `proxy_ip` = the prefill node's eno0 IP.

## Known Limitations (honest, open)
- **TP4/TP4 walls at 64K** — decode-side int32 kernel offset (M>8192 decode-GEMM). DP≥2 (EP8,
  TP4×DP2) escapes it; pure-TP needs the int32→int64 decode-GEMM fix (patches 15/16, WIP).
- **MTP over disagg: colocated only.** MTP + decode cudagraph work colocated (~17ms). Over
  disagg the connector group-symmetry is solved and it recalls to ~10K, but faults at >16K on
  a FlyDSL/asm spec-decode kernel int32 offset (`next_n=2`). 🚧 experimental, not production.
- **EP16 / WideEP (>1P>1D):** the base stack's amd_mori + aiter regress GLM DSA cross-node
  (GPU fault on first forward). 🚧 needs a MoRI+AITER source bump (see the GLM-5.1 WideEP recipe).
- **512K:** blocked on the same int32→int64 kernel track as TP4/TP4.
