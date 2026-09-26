# [vllm_disagg] GLM-5.3-Flash-FP8 disaggregated (1P/1D) recipe for MI300X/MI325X (gfx942)

## Summary
Adds a **gfx942** (AMD MI300X 192GB / MI325X 256GB) recipe for serving
**GLM-5.3-Flash-FP8** (`Glm5NextForConditionalGeneration` — FP8 E4M3, MLA + DeepSeek
sparse-indexer + KDA-linear hybrid) **disaggregated** (prefill/decode split) over the
**MoRIIO** RDMA KV connector. This is the **sibling of PR254** (gfx950 / MI355X): same
structure and doc discipline, different arch, and independently measured gfx942
numbers.

Unlike PR254's from-source compile, this recipe is an **OVERLAY on the pinned vLLM
ROCm nightly** — the nightly already ships `glm5next` natively + the MoRIIO connector +
a prebuilt aiter + mori, so the GLM-5.3 fixes are pure-Python `.py` overlays + one
unified diff over dist-packages (no vLLM/aiter/mori recompile). A wholesale PR254-style
overlay *regressed* this stack (garbage at 24 tokens) due to a connector API skew, so
only the block-routing idea was ported and re-implemented against the newer 4D packed
MLA cache API.

New dir: `scripts/vllm_dissag/glm53_flash_gfx942/` +
`docker/vllm_disagg_inference.glmv53flash.gfx942.amd.Dockerfile`.

## What this adds
- A self-contained **Dockerfile** (base pinned by digest + patches baked + a
  from-source `vllm-router` built in a throwaway stage + serve scripts).
- The verified **vLLM patch set** (`patches/core/` + `patches/moriio_kbpb_fix/`) with a
  manifest README.
- Per-topology **serve scripts** (`serve/`): EP8/EP8, TP4/TP4, the production router
  build+launch, and a fallback toy proxy.
- Docs: `README.md`, `RESULTS.md`, `PROVENANCE.md`, `TEST_PLAN.md`.

## Verified results (measured on-device 2026-09-23/24, coordinator-verified)
NIAH = needle-in-haystack, greedy temp 0, `/v1/completions`, over the live MoRIIO
disagg proxy/router. gfx942, MI300X 192GB (same recipe on MI325X 256GB).

**EP8/EP8 — PRODUCTION-READY ≤256K:**
- **NIAH 21/21 cells PASS**, 8K–256K, all needle depths (0.1/0.5/0.9). 256K =
  266,709 real prompt tokens; independently reproduced 3/3 on a relaunched pair.
- **Production `vllm-router` A/B vs the toy proxy** (live EP8 pair): 256K conc8 TTFT
  P90 **196→48s (4.1×)**, output tok/s 2.2→10.2 (4.6×); 256K conc16 229/396→90/94s
  P50/P90 (~4×, **no collapse**, 16/16 OK); 100K conc8 P90 40→22s (1.9×). TPOT stable
  ~30ms, recall intact through routing. The router removes the toy proxy's ~80%
  prefill-queuing.

**TP4/TP4 — clean to ~62K:**
- NIAH clean at all depths 4K–**62K**; the kbpb transfer fix **self-adapts** to TP4
  (factor 9) with zero code changes. Walls at **64K** on a decode-side memory-access
  fault (root-caused to the int32→int64 kernel workspace-offset overflow; block-size
  ruled out). MTP OFF on the TP4 decode leg (per-group block-count assert).

**TP4×DP2 — in progress** (routing/dpfix being validated; build+accuracy+perf pending).

Full grids + the before/after + serve config are in `RESULTS.md`.

## The fixes (why disagg is correct)
- **vLLM kpool slot-mapping** (`patches/core/14_*.patch`): expands each coarse shared
  hybrid-KV block into its fine DSA indexer pages so every sparse-indexer pool gets a
  unique monotonic slot (was collapsing ~7 pools onto one physical slot →
  nondeterministic long-context recall). The colocated recall fix.
- **MoRIIO per-group block expansion** (`patches/moriio_kbpb_fix/`): the DSA
  `indexer.k_cache` is paged at a finer kernel block (`kbpb`) than the shared group
  block, so each group block-id is expanded into its `kbpb` kernel sub-blocks before
  byte offsets (`kbpb` derived per-leg from the handshake metadata; kbpb=1 elsewhere is
  a byte-identical no-op). Without it, decode reads the wrong offset past ~4096 tokens →
  recall corruption. The disagg transfer fix.
- **core 01-07,11**: eager-serve correctness + decode CUDA graph (opaque-op so Dynamo
  never traces the indexer `@triton.jit` kernels) + flydsl fast prefill.
- **config-level (serve scripts, mandatory)**: `--max-num-batched-tokens 16384` +
  `--no-enable-prefix-caching` (both legs) = the long-context unlock;
  `VLLM_USE_BREAKABLE_CUDAGRAPH=1` + PIECEWISE for decode cudagraph; the aiter env
  block; the MoRIIO fabric env (atomic-MR strip, GID index).

## What's baked in the image
- Base `vllm/vllm-openai-rocm:nightly @ sha256:f169e8df…` (vLLM 0.3.1.dev3+g0bfc7a15d,
  torch 2.12.0, ROCm 7.2.3, aiter v0.1.19, mori v1.1.0) — prebuilt aiter/mori kept
  as-is.
- core 01-07,11 + patch14 (recall) + kbpb (disagg transfer), applied idempotently.
- `vllm-router` (Rust) built from source (upstream `0fb97775` + dpfix `82dc9811`) into
  `/usr/local/bin/vllm-router`.
- Serve scripts at `/opt/serve`. `/app/versions.txt` provenance line.
- **Not baked (by design, host-tunable in serve scripts)**: the config-level knobs
  above. **Do NOT set `AITER_JIT_DIR`** (would break the prebuilt aiter). See
  `PROVENANCE.md` for exact pins.

## Known limitations / open items
- **512K**: faults on the int32→int64 kernel workspace-offset overflow (fix is a
  separate track).
- **TP4/TP4 walls at 64K** on the same int32→int64 decode-GEMM class; verified clean to
  ~62K. EP8/EP8 clears it to 256K via DP8 per-rank geometry. **We do NOT claim TP4 256K
  or 512K.**
- **TP4×DP2**: in progress (routing being validated); no accuracy/perf yet.
- **MTP-over-disagg**: per-group block-count reconcile pending (MTP OFF on the TP4
  decode leg).
- **The Dockerfile has NOT been test-built** — assembled from the verified on-device
  flow; build once on a node with the base image + github/crates.io access. The router
  HEAD `5f3910f` is a local post-cherry-pick sha (the Dockerfile reproduces it from
  `UPSTREAM_REF` + `DPFIX_REF`). See `PROVENANCE.md`.

## Test plan
See `TEST_PLAN.md`. In brief, on two nodes (8×MI300X/MI325X gfx942, mlx5 RoCE) with the
built image:
- **Gate 0** — image self-containment (patches baked, `which vllm-router`, kbpb marker).
- **Gate 1** — EP8/EP8 bring-up (decode → prefill → router RDP=8) + short NIAH recall.
- **Gate 2** — EP8/EP8 long-context NIAH grid to 256K, all depths (over → clean HTTP-400).
- **Gate 3** — production router A/B vs toy proxy (TTFT @256K).
- **Gate 4** — TP4/TP4 bring-up (MTP off) + NIAH to ~62K (64K expected to fault).

## Files
- `docker/vllm_disagg_inference.glmv53flash.gfx942.amd.Dockerfile` (mirrored in the
  recipe dir as `Dockerfile`).
- `scripts/vllm_dissag/glm53_flash_gfx942/patches/` — the verified overlays + manifest.
- `scripts/vllm_dissag/glm53_flash_gfx942/serve/` — launchers + router.
- `scripts/vllm_dissag/glm53_flash_gfx942/{README,RESULTS,PROVENANCE,TEST_PLAN}.md`.

## Related
- **PR254** — the gfx950 (MI355X) sibling recipe this parallels.
