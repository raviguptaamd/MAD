# GLM-5.3-Flash-FP8 disaggregated (1P/1D) on MI355X (gfx950) + AMD AI NIC (ionic)

`GLM-5.3-Flash` (`Glm5NextForConditionalGeneration`), **FP8** (E4M3, dynamic
activation scaling), is a DeepSeek-DSA / MLA +
KDA-linear-attention hybrid. This recipe serves it **disaggregated (prefill /
decode split)** over the **MoRIIO** KV-transfer connector, with verified
needle-in-haystack recall (see `RESULTS.md`).

Two configurations are provided, **both served by one image** (they differ only
in launch env, not in the build):
- **TP4 1P/1D** — tensor-parallel 4 per leg, MoRIIO KV transfer only.
- **EP8 1P/1D** — DP8 + expert-parallel, MoRIIO KV transfer + allgather/reducescatter
  MoE dispatch.

---

## TL;DR — what this is and why it exists

Disaggregated inference splits prefill and decode onto separate GPUs/nodes and
ships the KV cache between them. On this stack the KV hop runs over **MoRIIO**
(mori's RDMA connector) on **ionic** NICs. Getting GLM-5.3-Flash to serve
disaggregated with **correct long-context recall** took two fixes:

1. a **real code bug** in the MoRIIO connector's MLA KV block-mapping (garbage
   recall — decode read zero KV), and
2. a **launch-time knob** (single-chunk prefill) so prompts longer than one
   prefill chunk transfer their whole KV.

With both applied: **exact needle recall to 100K tokens (TP4)** and **~28K
(EP8)**, all needle depths. Details below.

**Only one subcomponent changed to fix this: vLLM.** Everything else in the image
(mori, aiter, vllm-router, base) is pinned unchanged at the same refs the proven
GLM-5.1 disagg image used. See "What's in the image" below.

---

## The investigation (how we found the two fixes)

**Symptom.** Disagg served fluent *local* continuation but recall was garbage at
every length — e.g. a prompt stating "the password is banana … the password is"
decoded to "is is is is". Short greedy prompts ("The capital of France is" →
"Paris") looked fine, which masked it at first.

**Localization.** The *same image* run **colocated** (no prefill/decode split, no
MoRIIO) recalled correctly at long context. That isolated the bug to the
**MoRIIO KV transfer path**, not the model, kernels, or toolchain.

**Root cause (fix #1).** GLM-5.3-Flash uses MLA, so its KV cache is several
tensors (main attention `self_attn.attn`, the DSA `indexer.k_cache`), each paged
at its **own kernel block-size**. But vLLM's hybrid KV allocator hands out
block-ids in a padded **group** block unit (1152 tokens here). Each MLA tensor
therefore holds `num_group_blocks * kbpb` kernel blocks, where `kbpb`
("kernel-blocks-per-group-block") differs per tensor (`self_attn.attn` kbpb=18,
`indexer.k_cache` kbpb=9 for this model). The connector was transferring at the
**raw group block-id** — so the main attention KV was written to the wrong
(effectively empty) kernel blocks. Decode then read **zero** attention KV for
everything except the last group block → fluent-but-wrong output.

The fix expands each group block-id `N` into its `kbpb` contiguous kernel
sub-block-ids `[N*kbpb … N*kbpb+kbpb-1]`, for both the local and remote id lists,
for each MLA cache tensor at its own `kbpb`. `kbpb` is derived from the true
group-block count (`min` blocks over all kv_caches). Files:
`vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_layout.py`
(the `_mla_kernel_blocks_per_group_block` helper + the expansion in
`compute_block_transfer_offsets`) and `.../moriio_connector.py` (compute
`_num_group_blocks` once at `register_kv_caches` and pass it in).

**Second ceiling (fix #2).** With fix #1, recall was exact up to ~16K then broke.
Discriminator runs showed the connector transfers only the **final prefill
chunk's** KV (a known limitation in its own code). With the default chunk
(`max_num_batched_tokens` 16384) any prompt longer than that loses its earlier
chunks' KV on the decode side. Making prefill **single-chunk**
(`--max-num-batched-tokens >= max-model-len` on the prefill leg) transfers the
whole prompt's KV. Recall then held to 100K (TP4). This is a **launch arg, not
code** — documented per-leg in the orchestrators.

**Dead ends ruled out (so nobody re-chases them):** the DSA tail_cache was a red
herring (colocated recalled without it); fp4 GEMM was not the culprit
(instrumented and cleared); it is not a kernel/toolchain/image gap (colocated
proves the model is fine on gfx950).

---

## What makes disagg correct (the two fixes, in brief)

1. **MoRIIO MLA KV block-mapping fix** (vLLM code) — carried **in-source** by the
   pinned `VLLM_REF` in the Dockerfile. There is no runtime patcher. An image
   without it boots and serves but returns silently wrong long-range output.
2. **Single-chunk prefill** (launch arg) — `--max-num-batched-tokens >=
   --max-model-len` on the **prefill** leg. Verified exact recall to 100K (TP4).

## What's in the image (subcomponents)

| Subcomponent | Pin | Changed for GLM-5.3? |
|---|---|---|
| **vLLM** | `raviguptaamd/vllm @ 9a4642006` (branch `glm53-flash-moriio-mla-fix`) | **YES — the MLA KV fix** |
| mori | `ROCm/mori @ 624002c897a3` | no (same as proven GLM-5.1; already strips the ionic-rejected atomic-MR bit) |
| aiter | `raviguptaamd/aiter @ 624e43586b` | no |
| vllm-router | `raviguptaamd/router @ 82dc9811` | no |
| base image | `rocm/vllm-dev:ci_base-dedbf6be…` | no |
| arch target | `gfx950` (MI355X) | corrected from GLM-5.1's gfx942 |

Because the pinned mori already carries the atomic-MR strip (GLM-5.1 validated it
on ionic with **no** runtime overlay), the **image built from this Dockerfile is
self-contained** — it does not need the dev overlay set the debug image used.

## The 6 bring-up essentials

1. **Image**: build from `docker/vllm_disagg_inference.glmv53flash.ubuntu.amd.Dockerfile`
   (its `VLLM_REF` fork pin carries the vLLM connector fixes; its `MORI_REF` fork
   pin carries the ionic mori fixes; arch pinned gfx950). All fixes are in-source —
   no runtime overlays needed.
2. **VRAM headroom** — `GPU_MEMORY_UTILIZATION` 0.5 (TP4) / 0.40 (EP8). Too high
   starves the DSA-indexer Triton code-object load at long context
   (HSA_STATUS_ERROR_OUT_OF_RESOURCES). EP8 also needs
   `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=4096` (single-chunk indexer).
3. **glibc swap** (`GLIBC_SWAP=1` in the launcher) — fresh nodes ship glibc 2.39;
   the ionic RDMA driver needs `GLIBC_2.38`. The launcher bind-mounts host glibc
   at runtime; no rebuild.
4. **atomic-MR strip** — ionic rejects `REMOTE_ATOMIC` memory regions (errno
   14/22). This is **baked into the pinned mori** in the image build; no overlay
   needed. (The `MORI_PATCHED=1` launcher path exists only for running a
   host-built patched `.so` set on an image that lacks it.)
5. **The MLA KV fix** (fix #1 above) — carried by the pinned `VLLM_REF`.
6. **Single-chunk prefill** (fix #2 above) — `--max-num-batched-tokens` in the
   prefill leg's launch args.

## Bring-up ORDER (the launcher is order-sensitive)

- Router (`ROLE=proxy`) with **`ROUTER_DP_LOCAL=1` for TP** (`=8` for EP/DP8).
  The wrong value makes the router return HTTP 400 for every request.
- Bring the **decode leg up first** (to "Application startup complete"), **then
  prefill**. Prefill caches the decode mori handshake; if you restart decode
  after prefill is up you must restart prefill too, else requests hang.
- **Warm** with 2–3 tiny requests (mori CreateSession is a cold-start race; the
  first request may 503).

## Gold pairs (ionic rail routing — read before picking nodes)

MoRIIO stripes the KV write across all 8 ionic rails. A rail only carries traffic
if **both** nodes' NIC on that plane sit on the same `/64` subnet. A "gold" pair
= all 8 rails match → full bandwidth, zero flush, **no IBDEV pinning**. A
mismatched rail → RDMA transport-retry (CQE status=12) → the whole KV transfer
flushes and the request hangs/500s. To find a gold pair, GID-map both candidates:

```
for i in 0 1 2 3 4 5 6 7; do printf "ionic_%s: " $i; \
  cat /sys/class/infiniband/ionic_$i/ports/1/gids/1; done
```

Compare the 3rd hextet (subnet suffix) rail-by-rail: all 8 match → gold (no
IBDEV); some match → pin `IBDEV=ionic_<matching rail>` (single-rail, reduced BW);
none → pick another node.

## Run it

`vllm_pd_launch.sh` is the per-node, per-role launcher (serves the image as
built — no source overlays; the fix is in `VLLM_REF`). The two orchestrators
below drive it on both legs over `spur exec` (this cluster's per-node exec);
swap `drive()` for your own remote exec (ssh/srun) elsewhere.

TP4 1P/1D (prefill = node A, decode = node B):
```
PF_JOB=<A_handle> DC_JOB=<B_handle> PF_IP=<A_ip> DC_IP=<B_ip> \
IMG=<image built from the glmv53flash Dockerfile> \
MODEL=<GLM-5.3-Flash weights path on the nodes> \
ROUTER_BIN=<vllm-router binary path on the nodes> \
REMOTE_DIR=<path to this recipe dir on the nodes> \
INFRA_ENV="GLIBC_SWAP=1 HOSTLIBS=<dir>" \
  bash run_flash_disagg_tp4.sh
```
(`INFRA_ENV` is optional — only for fresh nodes needing the glibc-2.39 closure;
omit if your image already carries it.) It precleans, launches router + decode +
prefill in the correct order, warms the handshake, and prints a recall smoke test
(expect `DELTA-9931` at 500w and 8000w). `orch_ep8.sh` is the EP8 variant (same
env, `MODE=ep`, `ROUTER_DP_LOCAL=8`).

Alternatively, serve via MAD's standard disagg entry point using the registry:
`MODEL_NAME=GLM-5.3-Flash` selects the recipe block in `models.yaml` (see
`scripts/vllm_dissag/README.md` for the moriio.sh / run_xPyD_models.slurm flow).

---

## Sample test plan (reproduce + gate a PR)

A reviewer with two gold-pair nodes and the built image reproduces the whole
result in ~15 min per config. **One image, both configs.**

**Gate 0 — image is correct.** In the built image:
`cat /app/versions.txt` shows `VLLM_REF=9a4642006…`; the moriio files carry the
fix (`grep -R _mla_kernel_blocks_per_group_block` in the installed vLLM returns a
hit); arch is gfx950.

**Gate 1 — TP4 bring-up + short recall.** Pick a gold pair (GID-map above).
Run `run_flash_disagg_tp4.sh` with the env above. Expect both legs to reach
"Application startup complete", a tiny prompt to serve, and the built-in recall
smoke to print `DELTA-9931` at 500w **and** 8000w. This is the pass/fail gate:
8000w > one attention group block, so it exercises the block-mapping fix.

**Gate 2 — TP4 long-context needle.** Fire a ~40K-token prompt with the needle
at depth 0.9 (and 0.1/0.5/0.99); expect `DELTA-9931` at every depth. Optionally
push to 100K (util 0.5 has headroom). This gate exercises single-chunk prefill.

**Gate 3 — EP8 bring-up + recall.** On the second gold pair, `orch_ep8.sh`
(`MODE=ep`, `ROUTER_DP_LOCAL=8`, util 0.40, `SPARSE_IDX_MB=4096`, single-chunk).
Expect `DELTA-9931` at 8000w and up to ~28K (near the 32K max-model-len). Watch
for MoE all2all crashes (should be none with allgather/reducescatter) and indexer
OOM-resources (raise headroom / lower util if seen).

**Gate 4 — negative control (optional, proves the fix matters).** Run TP4 with
the fix disabled (an image whose `VLLM_REF` predates the fix, or bypass the
expansion): recall goes garbage at 8000w while short prompts still serve. This is
the "before" row in `RESULTS.md`.

**Expected results:** `RESULTS.md` — TP4 exact to 100K all depths; EP8 exact to
~28K all depths.

Smoke-only (no gold pair / single check): Gate 1's 8000w recall alone catches the
block-mapping regression; it is the single most valuable check.

---

## Model registry

- `scripts/vllm_dissag/models.yaml` — recipe block `GLM-5.3-Flash` (serving flags/env).
- `scripts/vllm_dissag/models.json` — CI entry `pyt_vllm_disagg_mori_glm-5.3-flash`.

## Files
- `vllm_pd_launch.sh` — per-node, per-role launcher (proxy / prefill / decode).
  Env-driven; no source overlays (fix is in the image). This is the piece both
  orchestrators call.
- `run_flash_disagg_tp4.sh` — TP4 1P/1D orchestrator (bring-up order + single-chunk knob).
- `orch_ep8.sh` — EP8 1P/1D orchestrator (DP8 + allgather/reducescatter MoE dispatch).
- `README.md` / `RESULTS.md` — this file + the verified NIAH recall matrix
  (TP4 → 100K, EP8 → ~28K).
