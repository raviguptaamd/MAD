# Kimi-K3 MXFP4 — vLLM 2P/2D disaggregated EP16 (cutting-edge stack)

Prefill/decode **disaggregated** Kimi-K3-MXFP4 serving across **4 MI300X/MI325X nodes**
on the **latest upstream stack**, with MoRI-EP all2all + the MoRIIO KV connector.

```
   PREFILL POOL                          DECODE POOL
  ┌───────────────┐   MoRIIO KV +      ┌───────────────┐
  │ 2 nodes = 16  │   KDA-state RDMA   │ 2 nodes = 16  │
  │ GPU, EP16     │ ────────────────►  │ GPU, EP16     │
  │ HT + eager    │   write + notify   │ LL + graph    │
  └───────────────┘                    └───────────────┘
```

## Stack (the "cutting-edge" upgrade vs PR #193)

| Component | Pin |
|---|---|
| Base image | `rocm/vllm-dev:ci_base-dedbf6be` |
| vLLM | upstream `d626108b` (2026-08-20) + folded K3 KDA/MoRIIO support |
| MoRI | `624002c8` |
| router | `82dc9811` (vllm-router, PD-disagg discovery mode) |
| AITER / flydsl | image-bundled (AITER a8w4 SiTUv2, flydsl 0.2.4) |
| WITH_NIXL | 0 |

## Formal per-role setup (the working contract)

| Role | all2all backend | cudagraph | KV role |
|---|---|---|---|
| **prefill** (master g25 + worker g32) | `mori_high_throughput` | `NONE` (eager) | kv_producer |
| **decode** (master g33 + worker g39) | `mori_low_latency` | `FULL_AND_PIECEWISE` | kv_consumer |

Per pool: **TP2 × DP8 → EP16** (no PP). Workers are headless native-vLLM DP members
(`--data-parallel-start-rank 4 --headless`); the router talks only to each pool's master.

## Load-bearing launch knobs (env)

- `MAX_MODEL_LEN` ≤ 320000 (512K blows the >5-min inductor-compile handshake — see below),
  `MAX_NUM_SEQS=32`, `MAX_NUM_BATCHED_TOKENS` ≤ 4096 (bigger blows the MoE profiling shape),
  `KV_CACHE_MEMORY_BYTES` (KV bytes pinned to skip the profile_run all2all deadlock).
- `THINKING_DEFAULT=false` → `--default-chat-template-kwargs '{"thinking":false}'`
  (K3 is a reasoning model; off for benchmarks so the needle lands in `content`).
- The launcher seds `HANDSHAKE_TIMEOUT_MINS` 5→30 in-container so long-ctx compile survives.

## Bring-up

`run_2p2d_launch.sh` deploys + starts workers→masters→router in the discovery window.
Start the router via `docker exec -d` (a late/foreground router misses discovery and the
engines' `_ping` threads exhaust `MAX_PING_RETRIES`). See `run_2p2d.sh` for the full contract.

## Status / known issue (tracked in `docs/PERF_DIAGNOSIS.md`)

- **Correctness: WORKS.** Single + concurrent requests recall correctly (NIAH 50K/100K PASS).
- **Open perf blocker: ~150–170s per-request stall.** After a fast, correctly-rank-matched
  handshake+notify (~2s), **both prefill and decode GPUs pin 100% with no logs for ~150s**,
  then correct output. Invariant across WRITE/READ mode and PIECEWISE/FULL_AND_PIECEWISE.
  Prime suspect: the new MoRI `624002c8` InterNodeV1LL decode all2all barrier on bnxt-Thor2
  (v3 used older MoRI + same `mori_low_latency` and got ~20s/50K). Investigation ongoing.

See `docs/PERF_DIAGNOSIS.md` for the full experiment log and `docs/OPT_ROADMAP.md` for
pending optimizations (decode MTP/speculative, prefill chunked-prefill + context parallel,
all2all backend A/B, MoRI bisect).
