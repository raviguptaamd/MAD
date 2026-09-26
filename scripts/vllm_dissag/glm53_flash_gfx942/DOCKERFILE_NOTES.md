# Dockerfile notes — GLM-5.3-Flash-FP8 gfx942 (MI300X/MI325X) disagg image

Companion to `glm53_recipe/Dockerfile`. What it bakes, what is left as ARG/TODO,
the one-command build + per-topology run, and assumptions to verify before build.

## What's baked (image layers)
- **Base**: `vllm/vllm-openai-rocm:nightly` (pin by DIGEST — see ARG note below).
  vLLM 0.3.1.dev3+g0bfc7a15d · torch 2.12.0 · ROCm 7.2.3 · aiter v0.1.19 · mori v1.1.0.
  The base's PREBUILT aiter + mori are kept as-is (overlay strategy; no recompile).
- **Core patches 01-07, 11** — applied via `apply_patches.sh` (idempotent, anchor-based).
  Eager-serve correctness + decode CUDA graph opaque-op + flydsl fast prefill.
- **Patch 14** (`14_kpool_slot_mapping.patch`) — THE recall fix, `patch -p1 --forward`,
  marker-guarded. Confirmed byte-identical to `serve/tp4/bundle/slotfix.patch`.
- **kbpb MoRIIO overlay** (`moriio_kbpb_fix/{layout,connector,common,engine}.py`) —
  disagg KV-transfer recall fix, file-copy over the native connector dir + pycache scrub.
  Confirmed byte-identical to the `serve/tp4/bundle/moriio_kbpb_fix/` copy.
- **vllm-router** — Rust binary, built in a throwaway stage-1 (toolchain never lands in
  the final image), installed to `/usr/local/bin/vllm-router`, `--help | grep moriio` gated.
- **Serve scripts** at `/opt/serve/` (whole `serve/` tree, incl. `tp4/`).
- **`/app/versions.txt`** — provenance: base digest + patch set + resolved router ref.

## What's config-level (in serve scripts, NOT image layers — by design)
Per STACK.md §3, these stay run-tunable so one image serves every topology/cluster:
`VLLM_USE_BREAKABLE_CUDAGRAPH=1`, `--compilation-config {"cudagraph_mode":"PIECEWISE"}`,
`--max-num-batched-tokens`, `--no-enable-prefix-caching`, MTP `--speculative-config`,
the AITER env block, and the mlx5 MoRIIO fabric env. **`AITER_JIT_DIR` is intentionally
NOT set** (would break the baked aiter).

## Left as ARG / TODO
| ARG | Default in file | Action before build |
|---|---|---|
| `BASE_DIGEST` | `sha256:f169e8df365b5112a1` (12-char stub) | Capture full 64-char digest from a live node and pass it. |
| `BASE_IMAGE` | `vllm/vllm-openai-rocm:nightly` | **To truly digest-pin, pass `--build-arg BASE_IMAGE=vllm/vllm-openai-rocm@sha256:<full>`** (see assumption 1). |
| `ROUTER_REF` | `PLACEHOLDER_dpfix_rebased_on_upstream_0fb97775` | **TODO**: replace with the confirmed rebased sha (raviguptaamd/router dpfix rebased on vllm-project/router main `0fb97775`) once that build is confirmed. Build FAILS on the placeholder — this is deliberate (no silent wrong-router). |
| `ROUTER_REPO` | `https://github.com/raviguptaamd/router.git` | Override if the rebase lands elsewhere. |
| `RUST_TOOLCHAIN` | `1.88.0` | Router deps (time/home) require rustc ≥1.88. |

## One-command build
```bash
docker build -f Dockerfile \
  --build-arg BASE_IMAGE=vllm/vllm-openai-rocm@sha256:f169e8df365b5112a1...<full-64> \
  --build-arg BASE_DIGEST=sha256:f169e8df365b5112a1...<full-64> \
  --build-arg ROUTER_REF=<rebased-dpfix-sha> \
  -t rocmshared/glm53-flash-disagg:gfx942-mi300x .
```

## Per-topology run (patches already baked — no cp/apply at runtime)
```bash
# long-lived container (weights hot)
docker run -d --name nite --network host --ipc host --privileged --group-add video \
  --device /dev/kfd --device /dev/dri --cap-add IPC_LOCK --shm-size 128G \
  --ulimit memlock=-1:-1 \
  -v /models/GLM-5.3-Flash-FP8:/models/GLM-5.3-Flash-FP8:ro \
  -v $HOST_CACHE:/opt/vllm_cache \
  --entrypoint bash rocmshared/glm53-flash-disagg:gfx942-mi300x -lc "sleep infinity"

# Colocated TP4 (1 node, 4 GPU) — PIECEWISE + MTP, recall to 100K+
docker exec -d nite bash /opt/serve/serve_piecewise.sh

# EP8/EP8 disagg (2 nodes × 8 GPU) — prefill EAGER, decode PIECEWISE+MTP+kbpb, ≤256K
docker exec -d nite-prefill bash /opt/serve/serve_disagg_ep8_prefill.sh   # kv_producer node
docker exec -d nite-decode  bash /opt/serve/serve_disagg_ep8_decode.sh    # kv_consumer node
docker exec -d nite-prefill bash /opt/serve/serve_disagg_proxy.sh         # toy proxy (or vllm-router)

# TP4/TP4 disagg — recall clean to ~62K (MTP OFF on decode leg: use *_nomtp)
docker exec -d nite bash /opt/serve/tp4/serve_disagg_tp4_prefill.sh
docker exec -d nite bash /opt/serve/tp4/serve_disagg_tp4_decode_nomtp.sh
docker exec -d nite bash /opt/serve/tp4/serve_disagg_tp4_proxy.sh
```

## Assumptions made (VERIFY before relying on the build)
1. **Digest pinning via BASE_IMAGE, not BASE_DIGEST.** A Dockerfile `FROM ${BASE_IMAGE}`
   cannot interpolate a separate digest ARG into `image@digest` form portably, so the
   real pin is done by passing `BASE_IMAGE=...@sha256:<full>`. `BASE_DIGEST` is kept only
   for the label/versions.txt provenance line. If you prefer a hard in-file pin, hardcode
   the full `image@sha256:...` in the `ARG BASE_IMAGE=` default once the digest is known.
2. **Router source = rebased dpfix.** The task states the real router is a rebase of
   raviguptaamd/router `dpfix` onto vllm-project/router main `0fb97775`, built separately
   and not yet confirmed. `ROUTER_REF` is a placeholder; the stage will not produce a
   binary until it's set to a real sha. The `--help | grep moriio` gate assumes the
   rebased router still exposes a `moriio` mention in help (PR254's did).
3. **patch14 applies against the nightly's `indexer.py`.** `patch -p1 --forward` from
   `dist-packages/` assumes the diff's `a/vllm/...` context lines match this base
   (they were generated against it). If a future base drifts, the marker-guard turns it
   into a skip, NOT a hard fail — so a silent miss is possible; the verify step (§5) will
   show `slotfix = 0`. Watch that line in build logs.
4. **MoRIIO connector dir exists in the base** at
   `.../vllm/distributed/kv_transfer/kv_connector/v1/moriio`. The kbpb RUN hard-fails if
   absent (intentional — disagg is unusable without it).
5. **`--max-num-batched-tokens` value is NOT baked** and is genuinely ambiguous across
   sources — flagged below for the coordinator.

## Gaps / ambiguities for the coordinator to resolve
- **MNBT discrepancy.** STACK.md §3 says `--max-num-batched-tokens 49152`; RESULTS.md §A(d)
  and the actual `serve_disagg_ep8_{prefill,decode}.sh` use `16384` (default `MNBT=16384`,
  env-overridable); `serve_piecewise.sh` uses `8192`. Since MNBT lives in the serve scripts
  (not the image), the Dockerfile bakes whatever the scripts currently say (16384 disagg /
  8192 colocated). **The scripts and STACK.md disagree — reconcile STACK.md's 49152 vs the
  scripts' 16384.** RESULTS.md (coordinator-verified on-device) supports 16384, so I did not
  touch the scripts; confirm STACK.md §3 is just stale.
- **Router ref is unconfirmed** (assumption 2) — the one true blocker to a clean build.
- **Base digest is a 12-char stub** everywhere in the repo; the full 64-char digest must be
  captured from a live node. I could not obtain it (no device access).
- **`apply_patches.sh` vs `apply_tp4_disagg.sh` scope.** The repo's top-level
  `apply_patches.sh` only does core 01-07,11 (no patch14, no kbpb). The FULL disagg flow
  lives in `serve/tp4/apply_tp4_disagg.sh`. I modeled the Dockerfile on the FULL flow
  (core → patch14 → kbpb), which is what STACK.md/RESULTS.md describe as the production
  disagg stack — but I split it into explicit RUN layers rather than calling
  apply_tp4_disagg.sh (that script expects `/tmp/tp4stage`, not `/opt/glm53/patches`, and
  drops a NIAH harness into /opt/vllm_cache). Confirm this layering matches intent.
- **Provisional patches 09/10/12 excluded** (matches apply_patches.sh + STACK.md). Confirm
  none are wanted in the production image.
- **verify step is non-fatal** by design (won't fail the build on a marker gap). If you want
  a hard gate, change the §5 RUN to `exit 1` on `slotfix/kbpb == 0`.
```

## Build requirement: raise nofile ulimit (cargo router stage)
The router-build stage runs `cargo build --release` which opens many file descriptors in
parallel. On hosts with a low default docker-build nofile limit this fails with
`cargo ... Too many open files (os error 24)` (exit 101). Build with:
```
docker build --ulimit nofile=1048576:1048576 --pull -t <tag> .
```
This is a build-host limit, NOT a recipe defect — the patches/overlay stages are unaffected.
