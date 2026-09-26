# GLM-5.3-Flash-FP8 disaggregated — verified recall + perf results (MI300X/MI325X gfx942 + MoRIIO)

Per-topology accuracy + perf, with the exact reproducible serve config. All numbers
measured on-device (2026-09-23/24) and **coordinator-verified** (not agent-reported).
gfx942 = MI300X (192GB) AND MI325X (256GB) — same recipe both SKUs.

Needle-in-haystack (NIAH): a code needle (`74923`) is inserted at a given depth into
prose filler; the model is asked to recall it. Greedy (temperature 0),
`/v1/completions`, MoRIIO KV transfer, over the live disagg proxy/router.

---

## TOPOLOGY A — EP8/EP8 disaggregated (1 prefill node × 8 GPU, 1 decode node × 8 GPU)  ✅ PRODUCTION-READY ≤256K

### (a) NIAH accuracy grid (length × needle depth), deterministic temp 0, over the live disagg proxy

| Length | depth 0.1 | depth 0.5 | depth 0.9 |
|--------|-----------|-----------|-----------|
| 8K     | PASS | PASS | PASS |
| 32K    | PASS | PASS | PASS |
| 64K    | PASS | PASS | PASS |
| 100K   | PASS | PASS | PASS |
| 128K   | PASS | PASS | PASS |
| 190K   | PASS | PASS | PASS |
| 256K (266,709 real prompt tok) | PASS | PASS | PASS |

**21/21 cells PASS** — clean across the full 8K–256K envelope at all needle depths
(edges + middle). 256K independently coordinator-reproduced 3/3 @ 258,958 ptok on a
relaunched pair. Ceiling is `max_model_len` (270000): a prompt above it is cleanly
rejected (HTTP 400), not mis-recalled.

### (b) Perf — PRODUCTION vllm-router (prefill-aware). Supersedes the toy-proxy numbers.

| ISL | conc | TTFT P50 | TTFT P90 | mean TPOT | ok-rate | vs toy proxy |
|-----|------|----------|----------|-----------|---------|--------------|
| 100K | 8  | 21.6s | 21.7s | 30.8ms | 8/8   | P90 40→22s (1.9×) |
| 256K | 8  | 48.4s | 48.4s | 30.0ms | 8/8   | P90 196→48s (**4.1×**) |
| 256K | 16 | 90.2s | 94.0s | ~30ms  | 16/16 | 229/396→90/94s (~4×, no collapse) |

- **A/B measured live** (2026-09-24) on the EP8/EP8 pair: toy proxy vs production
  `vllm-router`, same legs. Recall intact through routing (256K NIAH PASS via router).
- Additional throughput at 256K conc8: output **10.2 tok/s** on the router vs 2.2 on
  the toy proxy (4.6×).
- The router's prefill-aware scheduling removes the ~80% prefill-queuing that made the
  toy proxy's TTFT collapse at concurrency. Router TTFT is tight (P50≈P90) where the
  toy proxy showed wide serialized spread (256K/16 toy P50/P90 229.7/396.3s).
- MTP acceptance confirmed (chunk-count < completion_tokens). Decode TPOT stable ~30ms.
- The remaining TTFT floor is **raw prefill compute** (measured conc1 256K prefill
  floor = 28.4s) — addressable only by chunked-prefill / TP-prefill tradeoffs, not
  routing.

Toy-proxy baseline (deprecated, kept for reference): 100K/8 TTFT P90 40.2s; 256K/8
P90 196.2s; 256K/16 P50/P90 229.7/396.3s.

### (c) VRAM / KV envelope
gpu_memory_utilization 0.6 (prefill), DP8 + expert-parallel, block-size 4. 256K/conc16
fit with no OOM.

### (d) Exact reproducible serve config (both legs)
```
--max-model-len 270000  --max-num-batched-tokens 16384  --no-enable-prefix-caching
--data-parallel-size 8  --enable-expert-parallel  --all2all-backend allgather_reducescatter
--block-size 4  --kv-transfer-config MoRIIOConnector (kv_producer / kv_consumer)
Prefill: --enforce-eager
Decode:  --compilation-config '{"cudagraph_mode":"PIECEWISE"}'  VLLM_USE_BREAKABLE_CUDAGRAPH=1
         --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
Patches: core 01-07,11 + patch14 (recall) + moriio_kbpb_fix/ (per-group block expansion, kbpb=17)
```
Serve scripts: `serve/serve_disagg_ep8_{prefill,decode}.sh` + `serve/router/serve_router.sh` (TOPO=ep8).

### (e) Recommendation
EP8/EP8 is the **lead production config for long-context + throughput** to 256K. Ship
for ≤256K workloads on the production `vllm-router` (TTFT is the lever, not the model).
512K needs the int32→int64 kernel offset fix (separate track — see open items).

---

## TOPOLOGY B — TP4/TP4 disaggregated (prefill TP4 eager, decode TP4 PIECEWISE)  ✅ recall clean to ~62K

**The kbpb transfer fix SELF-ADAPTED to TP4 with zero code changes** — disproving the
prior assumption that TP4 needed a bespoke kbpb variant. The fix is geometry-derived:
TP4 shards the DSA `indexer.k_cache` to factor **9** (decode nb=17739/ref=1971, prefill
nb=21627/ref=2403; vs EP8's 17); both sides derive the SAME factor so paired block
lists stay aligned. Fires only on `indexer.k_cache` layers (all others kbpb=1 no-op,
verified byte-identical across block-size 4/16/32).

### NIAH ceiling (needle 74923, over the TP4 proxy)
| Length | d0.1 | d0.5 | d0.9 |
|--------|------|------|------|
| 4K–32K | PASS | PASS | PASS |
| 48K / 56K / 60K | – | PASS | – |
| **62K (last clean)** | PASS | PASS | PASS |
| 64K | – | **FAULT** | – |
| 100K | – | FAULT (500) | – |

**Ceiling ~62K, walls at 64K.** The 64K wall is NOT a transfer/kbpb bug (32K–62K all
clean through the same transfer) — it is a **decode-side hard fault**: "Memory access
fault by GPU node-{6,7,8,9}" (all 4 decode GPUs) → EngineDead; prefill survives. Not
OOM (85GB/GPU, KV 0%). Root-caused to the **int32→int64 kernel workspace-offset
overflow** (M>8192 decode-GEMM class) — block-size/kpool alignment was tested and
RULED OUT (all of block-size 4/16/32 fault at exactly 64K with byte-identical kbpb
params). EP8/EP8 clears this to 256K via DP8 per-rank-workspace geometry (smaller
per-rank offset); TP4 (4-GPU, less headroom) needs the int32→int64 kernel fix (shared
with the 512K track) to lift the ceiling.

**MTP must be OFF on the TP4 disagg decode leg** — it crashes EngineCore on the first
request (per-group block-count assert, `moriio_connector.py:766`; the MTP-over-disagg
reconcile is an open item). Use `serve/tp4/serve_disagg_tp4_decode_nomtp.sh`.

Perf sweep for TP4 is **pending** (deferred until the 64K decode-fault is tuned/fixed
to lift the ceiling).

## TOPOLOGY C — TP4×DP2 / TP4×DP2  ✅ ROUTING PROVEN, recall to 260K (perf sweep in flight)
2 DP ranks per leg (8 GPU/leg), routed by vllm-router with `--intra-node-data-parallel-size 2
--moriio-dp-size 2` (the `dpfix` — load-bearing for DP≥2). **First DP≥2 run of this stack; it works.**
- **Routing:** both DP ranks of each leg registered + carried traffic under round-robin; router logged
  "DP-aware mode enabled (intra_node_data_parallel_size=2)". No wedge, no "remote blocks never arrived".
- **Recall: NIAH 21/21 PASS**, 8K→256K (240K tok) at ALL depths (0.1/0.5/0.9). All three 64K depths pass.
- **KEY FINDING:** TP4×DP2 does NOT hit the ~64K decode int32 fault that pure TP4/TP4 does. DP≥2 halves
  the per-DP-rank workspace, keeping the overflowing offset under 2^31 — the same escape as EP8's DP8.
  **So DP≥2 is itself a mitigation for the 64K fault, independent of the kernel patch.**
- **Perf** (streaming, input/output tok/s; TPOT flat ~27-28ms; all runs full ok-rate):

  | ISL,conc | TTFT P50 | TTFT P90 | in tok/s | out tok/s | prefill tok/s/req | TPOT |
  |---|---|---|---|---|---|---|
  | 100K,1 | 8.7s | 8.7s | 8190 | 10.5 | 11477 | 27.5ms |
  | 100K,8 | 27.6s | 36.1s | 19838 | 25.4 | 4935 | 27.8ms |
  | 100K,16 | 44.7s | 70.3s | 21296 | 27.3 | 2670 | 28.1ms |
  | 256K,1 | **25.8s** | 25.8s | 8200 | 4.4 | 9303 | 27.3ms |
  | 256K,8 | 78.7s | 106.2s | 17298 | 9.2 | 4287 | 28.0ms |
  | 256K,16 | 134.4s | 213.6s | 17632 | 9.4 | 2202 | 28.2ms |

- **TP-sharded prefill beats DP-replicated:** TP4×DP2 conc1 256K TTFT **25.8s vs EP8's 28.4s** (~9% lower) —
  4-way attention sharding per DP rank → 9303 prefill tok/s/req. Hybrid-value datapoint for TTFT reduction.
- **Enabling bug fixed:** `get_port_offset` at 4 sites omitted `tp_size` → co-located DP ranks collided on
  ZMQ ports at TP4×DP2 (harmless at EP8 tp_size=1). Fixed in the connector overlay.
- MTP on the decode leg: reconcile fix written (skip MTP's trailing decode-only KV group); device-test pending.

---

## Before the fix (for reference)
- **Without the MoRIIO per-group block expansion (kbpb):** disagg served fluent local
  continuation but recall corrupted beyond one attention block (~4096 tokens) — the
  identical prompt sent DIRECT to the prefill leg (no transfer) recalled fine, which
  localized the bug to the KV transfer. Before: 3379 tok PASS, 4489 tok FAIL; after:
  4489/6709/8929/17819/35599/71159 tok ALL PASS, kbpb confirmed firing only on
  `indexer.k_cache`.
- **Without patch14 (kpool slot-mapping):** ~7 DSA pools collapsed onto one physical
  slot → nondeterministic long-context recall even colocated.

## Open items (not blocking ≤256K EP8/EP8 production)
- **512K**: int32→int64 kernel offset fix (shared root cause with TP4 64K).
- **TP4/TP4 64K decode fault**: the int32→int64 decode-GEMM fix lifts TP4 past ~62K.
- **TP4×DP2**: build + accuracy + perf (routing being validated).
- **MTP-over-disagg**: per-group block-count reconcile (for the TP4 decode leg).

## Environment
- Base image: `vllm/vllm-openai-rocm:nightly @ sha256:f169e8df…` (vLLM
  0.3.1.dev3+g0bfc7a15d, torch 2.12.0, ROCm 7.2.3, aiter v0.1.19, mori v1.1.0) +
  `patches/` overlays. gfx942.
- Nodes: 8-GPU MI300X (gfx942), mlx5 RoCEv2 (`MORI_IB_GID_INDEX=3`, atomic-MR strip),
  one leg per node.
- Model: `GLM-5.3-Flash` (`Glm5NextForConditionalGeneration`), FP8.
