# GLM-5.3-Flash disagg (gfx942) — test & benchmark plan

Reproduce and gate the recipe on **two nodes** (8×MI300X 192GB or MI325X 256GB, gfx942,
mlx5 RoCEv2 each). Image built from `Dockerfile` (base pinned by digest — see
`PROVENANCE.md`). Patches are **baked** (no runtime overlay mounts). The config-level
knobs live in the serve scripts.

## Prerequisites
- Two gfx942 nodes, one leg each; mlx5 RoCE rails reachable between them
  (`MORI_IB_GID_INDEX=3`). Model weights at `/models/GLM-5.3-Flash-FP8` on both.
- Long-lived container per node (see `README.md` `docker run`), host cache mounted at
  `/opt/vllm_cache`.

---

## Part A — Correctness / boot gates (pass/fail)

**Gate 0 — image self-containment.** `docker inspect` shows the patches baked (no
`patches/*.py` bind-mounts), arch gfx942. In-image checks:
`grep -c GLM53_KPOOL_SLOT_MAPPING_FIX …/mla/indexer.py` = 1;
`grep -c GLM53_INDEXER_KBPB …/moriio/moriio_layout.py` = 1;
`which vllm-router` = `/usr/local/bin/vllm-router`; `cat /app/versions.txt`.

**Gate 1 — EP8/EP8 bring-up + short recall.** Decode leg first
(`serve/serve_disagg_ep8_decode.sh`), then prefill
(`serve/serve_disagg_ep8_prefill.sh`), then the router
(`TOPO=ep8 serve/router/serve_router.sh`, RDP=8). Both legs reach "Application startup
complete"; warm 2–3 tiny requests; a `74923` needle at 8K recalls. This is the single
most valuable check (8K > one attention group block → exercises the kbpb block-mapping
fix). RDP must = 8 (wrong → HTTP 400).

**Gate 2 — EP8/EP8 long-context NIAH grid.** `--max-model-len 270000
--max-num-batched-tokens 16384 --no-enable-prefix-caching` (both legs). Needle at
depths 0.1/0.5/0.9, lengths {8K,32K,64K,100K,128K,190K,256K}. Target: needle PASS at
every cell. Prompt > `max_model_len` → clean HTTP-400, not mis-recall. Exercises
patch14 + kbpb.

**Gate 3 — production router A/B.** With Gate-2 legs up, A/B the production
`vllm-router` vs the toy proxy (`serve/debug_toyproxy/`) at 256K conc8/16. Target:
router improves TTFT at high concurrency and holds recall through routing (measure).

**Gate 4 — TP4/TP4 bring-up + recall ceiling.** `serve/tp4/serve_disagg_tp4_prefill.sh`
+ `serve/tp4/serve_disagg_tp4_decode_nomtp.sh` (**MTP OFF**) +
`TOPO=tp4 serve/router/serve_router.sh` (RDP=1). Expect NIAH PASS 4K–62K all depths;
**64K is expected to FAULT** (decode-side memory-access fault — the documented int32
ceiling), so it is a *known-wall* check, not a regression.

### NIAH recall matrix (the correctness deliverable)
Needle `74923` in prose filler, `/v1/completions`, temp 0.

| topology | context | depths | pass criterion |
|---|---|---|---|
| EP8/EP8 | 8K,32K,64K,100K,128K,190K,256K | 0.1/0.5/0.9 | exact needle every cell (target) |
| TP4/TP4 | 4K…62K | 0.1/0.5/0.9 | exact needle to 62K; 64K known-fault |

---

## Part B — Performance benchmarks

**Metrics:** TTFT, TPOT, output tok/s. Report P50/P90 where a distribution exists.
Temp 0, fixed output length. Warm the JIT + handshake first (discard first 2–3 reqs).
Record image digest + gold pair + date per result. Driver:
`serve/…/perf_sweep_async.py` (async harness firing N concurrent fixed-length requests)
or vLLM `benchmark_serving` against the router `/v1/completions`.

**EP8/EP8 (measured — the reference):**

| ISL | conc | TTFT P50 | TTFT P90 | TPOT | out tok/s |
|---|---|---|---|---|---|
| 100K | 8 | 21.6s | 21.7s | 30.8ms | 21.7 |
| 256K | 8 | 48.4s | 48.4s | 30.0ms | 10.2 |
| 256K | 16 | 90.2s | 94.0s | ~30ms | — |

Reproduce these on the production router; they supersede the toy-proxy baseline.

**Sweep to extend (both EP8 and TP4≤62K):** contexts {1K,8K,60K,200K} × concurrency
{1,8}, output 128 — TTFT/TPOT/throughput table. Interesting deltas: EP8 vs TP4 (where
DP8+EP wins on throughput and what TTFT it costs); disagg vs colocated at 8K/60K.

**TP4 perf is pending** — deferred until the 64K decode-fault is fixed to lift the
ceiling (see `RESULTS.md` open items).

**Output:** a results table appended to `RESULTS.md` (image digest, env, node pair,
date), one row per cell.
