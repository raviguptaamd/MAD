# GLM-5.3-Flash-FP8 disaggregated (1P/1D) on MI300X / MI325X (gfx942) + MoRIIO

`GLM-5.3-Flash` (`Glm5NextForConditionalGeneration`), **FP8** (E4M3, dynamic
activation scaling), is a DeepSeek-DSA / MLA + KDA-linear-attention hybrid. This
recipe serves it **disaggregated (prefill / decode split)** over the **MoRIIO**
KV-transfer connector on **AMD MI300X (192GB) / MI325X (256GB) — gfx942** (same
recipe both SKUs), with verified needle-in-haystack recall (see `RESULTS.md`).

This is the **gfx942 sibling** of ROCm/MAD PR254 (gfx950 / MI355X). Same structure
and discipline; different arch, different delivery strategy (**overlay on the vLLM
ROCm nightly**, not from-source compile — see below), and independently measured
gfx942 numbers.

Two topologies are provided, **both served by one image** (they differ only in
launch env, not in the build):
- **EP8 1P/1D** — DP8 + expert-parallel per leg, MoRIIO KV transfer +
  allgather/reducescatter MoE dispatch. **Lead production config, verified to 256K.**
- **TP4 1P/1D** — tensor-parallel 4 per leg, MoRIIO KV transfer only. **Verified
  clean to ~62K** (decode-side kernel fault walls it at 64K — see RESULTS/limitations).

---

## TL;DR — what this is and why it exists

Disaggregated inference splits prefill and decode onto separate GPUs/nodes and ships
the KV cache between them. On this stack the KV hop runs over **MoRIIO** (mori's RDMA
connector) on **mlx5** RoCEv2 rails. Getting GLM-5.3-Flash to serve disaggregated
with **correct long-context recall** on gfx942 took two vLLM fixes:

1. the **kpool slot-mapping** fix (`patches/core/14_*.patch`) — expands each coarse
   shared hybrid-KV block into its fine DSA indexer pages so every sparse-indexer
   pool gets a unique monotonic slot (colocated recall correctness), and
2. the **MoRIIO per-group block expansion** (`patches/moriio_kbpb_fix/`) — the DSA
   `indexer.k_cache` is paged at a finer kernel block (`kbpb`) than the shared group
   block, so each group block-id must be expanded into its `kbpb` kernel sub-blocks
   before byte offsets, or the transfer reads the wrong offset past the ~4096-token
   wall → recall corruption.

With both applied, plus the config-level long-context unlock (`--max-num-batched-tokens
16384 --no-enable-prefix-caching`, both legs): **exact needle recall to 256K tokens,
all depths, EP8/EP8** (21/21 NIAH cells). Details below.

### How this recipe is delivered (READ THIS — it differs from PR254)

PR254 (gfx950) compiles vLLM/aiter/mori from source. **This gfx942 recipe is an
OVERLAY** on a pinned vLLM ROCm **nightly** that already ships `glm5next` natively +
the MoRIIO connector + a **prebuilt** aiter (`module_aiter_core.so`) + mori. Our
GLM-5.3 fixes are therefore pure-Python `.py` overlays + one unified diff, applied
idempotently over dist-packages — **no vLLM/aiter/mori recompile**. This matters:

- The nightly's prebuilt aiter/mori are the *proven* artifacts on gfx942. A wholesale
  PR254-style overlay **regressed** this stack (garbage at 24 tokens) because of a
  connector API skew (PR254's older 3D-cache API vs our newer 4D packed MLA caches).
  Only the block-routing *idea* was ported. See `patches/moriio_kbpb_fix/README.md`.
- **Do NOT set `AITER_JIT_DIR`.** The base ships a prebuilt `module_aiter_core.so`;
  pointing aiter at an empty dir forces a rebuild that fails. Persist only
  triton/vllm caches under the `/opt/vllm_cache` host mount (the serve scripts do this).

The `Dockerfile` bakes the whole flow as **one image**: (a) the pinned base by
digest, (b) core patches `01-07,11` + patch14 recall + the kbpb disagg overlay,
(c) a from-source `vllm-router` (Rust) built in a throwaway stage, (d) the
per-topology serve scripts at `/opt/serve`. It reproduces the ad-hoc "pinned image +
`docker cp patches` + serve" flow. **NOTE: the Dockerfile has NOT yet been
test-built** (assembled from the verified on-device flow; see `PROVENANCE.md` and the
Dockerfile header for the two build-args to fill).

### What's in the stack (subcomponents)

| Subcomponent | Pin | GLM-5.3 fixes delivered via |
|---|---|---|
| **base image** | `vllm/vllm-openai-rocm:nightly @ sha256:f169e8df…` (2026-09-17) | — (ships glm5next + MoRIIO + prebuilt aiter + mori) |
| **vLLM** | base image's vLLM `0.3.1.dev3+g0bfc7a15d` **+ `patches/` overlays** | core 01-07,11 (eager + decode cudagraph) + patch14 (recall) + kbpb (disagg transfer) |
| **aiter** | base image's aiter `v0.1.19` (prebuilt, kept as-is) | — (no aiter overlay; overlay strategy keeps the proven prebuilt) |
| **mori** | base image's mori `v1.1.0` (prebuilt, kept as-is) | — |
| **torch / ROCm** | torch `2.12.0` / ROCm `7.2.3` | — |
| **vllm-router** | upstream `vllm-project/router @ 0fb97775` + dpfix `raviguptaamd/router @ 82dc9811` → HEAD `5f3910f` | built from source into `/usr/local/bin/vllm-router` |
| arch target | `gfx942` (MI300X 192GB / MI325X 256GB) | — |

See `PROVENANCE.md` for exact pins/digests.

## The MANDATORY config-level fixes (serve flags/env — NOT patches)

These live in the serve scripts (so one image serves every topology) and are
load-bearing:

| item | value | why |
|---|---|---|
| `VLLM_USE_BREAKABLE_CUDAGRAPH=1` | env | decode cudagraph correctness (indexer runs outside the graph) |
| `--compilation-config '{"cudagraph_mode":"PIECEWISE"}'` | decode | decode CUDA graph |
| `--max-num-batched-tokens 16384` | both legs | **long-context unlock** — caps per-pass tokens below the int32 workspace-offset kernel ceiling. 16384 is the VERIFIED value that clears 256K EP8/EP8. |
| `--no-enable-prefix-caching` | both legs | prefix-cache partial re-prefill faults at 56–64K |
| `--speculative-config '{"method":"mtp","num_speculative_tokens":1}'` | decode (EP8) | MTP → TPOT ~30ms. **MTP must be OFF on the TP4 decode leg** (per-group block-count assert — see limitations). |
| aiter env | `VLLM_ROCM_USE_AITER=1` (+`_MLA/_MOE/_RMSNORM=1`, `_FP8BMM=0`), `VLLM_USE_DEEP_GEMM=0`, `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=4096` | kernel selection |
| MoRIIO fabric | `MORI_IO_DISABLE_ATOMIC_MR=1`, `MORI_IB_GID_INDEX=3`, `RDMA_DEVICES=mlx5_*` | RDMA transport env |

## Bring-up ORDER (order-sensitive)

1. **Decode leg first** (to "Application startup complete"), **then prefill**.
   Prefill caches the decode mori handshake; if you restart decode after prefill is
   up you must restart prefill too, else requests hang.
2. **Router last** (`serve/router/serve_router.sh`) — legs auto-register with the
   discovery ZMQ (`:36367`) via their ping threads; no leg restart needed if the port
   matches. `RDP` (`--intra-node-data-parallel-size`) MUST match the legs'
   `--data-parallel-size` (8 EP8 / 1 TP4 / 2 TP4×DP2) or every request 400s.
3. **Warm** with 2–3 tiny requests (mori CreateSession is a cold-start race; the
   first request may 503).

## Run it

The `serve/` scripts are per-node, per-role launchers (they carry the config-level
fixes above; edit the `proxy_ip` / model path for your nodes). One long-lived
container per node, then exec the role script:

```bash
# long-lived container (weights hot), per node
docker run -d --name nite --network host --ipc host --privileged --group-add video \
  --device /dev/kfd --device /dev/dri --cap-add IPC_LOCK --shm-size 128G \
  --ulimit memlock=-1:-1 \
  -v /models/GLM-5.3-Flash-FP8:/models/GLM-5.3-Flash-FP8:ro \
  -v $HOST_CACHE:/opt/vllm_cache \
  --entrypoint bash rocmshared/glm53-flash-disagg:gfx942-mi300x -lc "sleep infinity"
```

### EP8/EP8 — the lead config (verified to 256K)

```bash
# node B (decode / kv_consumer) FIRST
docker exec -d nite-decode  bash /opt/serve/serve_disagg_ep8_decode.sh
# node A (prefill / kv_producer)
docker exec -d nite-prefill bash /opt/serve/serve_disagg_ep8_prefill.sh
# node A — production router (RDP=8)
docker exec -d nite-prefill env TOPO=ep8 bash /opt/serve/router/serve_router.sh
```

Both legs: `--data-parallel-size 8 --enable-expert-parallel --all2all-backend
allgather_reducescatter --max-model-len 270000 --max-num-batched-tokens 16384
--no-enable-prefix-caching --block-size 4`, prefill `--enforce-eager`, decode
PIECEWISE + MTP. Recalls exactly to **256K** (see `RESULTS.md`).

### TP4/TP4 — verified clean to ~62K

```bash
docker exec -d nite bash /opt/serve/tp4/serve_disagg_tp4_prefill.sh
docker exec -d nite bash /opt/serve/tp4/serve_disagg_tp4_decode_nomtp.sh   # MTP OFF on decode
docker exec -d nite env TOPO=tp4 bash /opt/serve/router/serve_router.sh    # RDP=1
```

The kbpb transfer fix **self-adapts to TP4 with zero code changes** (factor 9 instead
of EP8's 17 — geometry-derived, both sides derive the same factor). Ceiling ~62K; a
decode-side memory-access fault walls it at 64K (see limitations). Use the `_nomtp`
decode script — MTP over the disagg decode leg crashes EngineCore on the first request.

### TP4×DP2 — in progress

```bash
docker exec -d nite env TOPO=tp4xdp2 MORIIO_DP_SIZE=2 bash /opt/serve/router/serve_router.sh   # RDP=2 + dpfix
```

The router `dpfix` (`--moriio-dp-size`) is load-bearing for DP≥2: plain upstream
re-hashes the KV-notify target as `blake2s(request_id) % dp_size` ≠ the routed prefill
rank → decode notifies the wrong rank → wedge. The dpfix forces both legs to honor the
routed rank verbatim. Build + accuracy + perf are **pending** (routing being validated).

## Files

- `Dockerfile` — the self-contained overlay image (base by digest + patches +
  from-source router + serve scripts). Also mirrored at
  `docker/vllm_disagg_inference.glmv53flash.gfx942.amd.Dockerfile`. **Not yet test-built.**
- `patches/` — `core/` (eager + decode cudagraph + patch14 recall) + `moriio_kbpb_fix/`
  (disagg KV-transfer fix) + a manifest README. Baked by the Dockerfile.
- `serve/` — per-topology launchers: `serve_disagg_ep8_*.sh`, `tp4/`, `router/`
  (production `vllm-router` build + launch), `debug_toyproxy/` (fallback proxy).
- `README.md` / `RESULTS.md` / `PR_BODY.md` / `PROVENANCE.md` — this file + the
  verified NIAH/perf envelope + the submittable PR description + the exact pins.
