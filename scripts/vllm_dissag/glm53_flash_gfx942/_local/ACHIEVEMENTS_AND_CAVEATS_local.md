# GLM-5.3-Flash-FP8 on AMD MI300X/MI325X (gfx942) — Achievements, Caveats, Recipes & Configs

Complete, honest record for the fresh ROCm/MAD PR. Everything below is **measured on-device**
unless marked PENDING. gfx942 = MI300X (192GB) AND MI325X (256GB) — identical recipe. Sibling to
PR254 (gfx950/MI355X); this is the gfx942 track with independently-measured numbers.

Model: `Glm5NextForConditionalGeneration` (native `glm5_next`), FP8 E4M3, 1M ctx. Hybrid:
KDA linear-attn + DeepSeek sparse-attn (DSA indexer, index_topk=2048, index_kpool=4) + MLA + MoE
(288 routed + 1 shared) + MTP (1 draft layer).

---

## 1. ACHIEVEMENTS (measured)

### ✅ EP8/EP8 disaggregated → 256K  — PRODUCTION
- NIAH **21/21** all depths (0.1/0.5/0.9), 8K → 256K (266,709 real prompt tokens). Coordinator-reproduced 3/3.
- Prefill EAGER (DP8+expert-parallel) + decode CUDA graph PIECEWISE, over MoRIIO KV transfer + vllm-router.
- Perf (production router): 256K conc8 TTFT **48s P90** (was 196s on the toy proxy = **4.1× faster**);
  conc16 90/94s (no collapse); TPOT ~30ms; input/output tok/s recorded.

### ✅ TP4×DP2 disaggregated → 256K  — PRODUCTION (first-ever DP≥2 run of this stack)
- NIAH **21/21** all depths to 256K (240K tok). Both DP ranks carry traffic, round-robin, no wedge.
- **DP≥2 mitigates the 64K int32 decode fault** (see caveats): recalls clean where pure TP4/TP4 faults at 64K.
- Perf (conc1 256K TTFT **25.8s** vs EP8 28.4s): **TP-sharded prefill is ~9% FASTER** — 9303 prefill tok/s/req.
  TPOT flat ~27-28ms across load; input tok/s saturates ~17-21K.

### ✅ Production vllm-router (real Rust router; toy proxy retired)
- Built = upstream `vllm-project/router` @0fb97775 + user dpfix `raviguptaamd/router` @82dc9811 → HEAD 5f3910f,
  vllm-router 0.1.15. The dpfix (`--moriio-dp-size`, remote_dp_rank_override) is **load-bearing for DP≥2**
  (plain upstream re-hashes the KV-notify target and wedges on DP≥2).
- Prefill-aware scheduling removes the ~80% prefill-queuing the toy proxy suffered. RDP = 8 (EP8) / 1 (TP4) / 2 (DP2).

### ✅ TP4/TP4 disaggregated → ~62K
- kbpb transfer fix **self-adapts** to TP4 (factor 9 vs EP8's 17) with **zero code change** — the
  "TP4 needs a bespoke kbpb" assumption was DISPROVEN.

### ✅ Decode CUDA graph FULL_AND_PIECEWISE on MI300X
- Captures clean (5.81 GiB, 19s), serves, NIAH 8K/64K PASS, **TPOT ~15.8ms vs PIECEWISE ~27ms = 1.7× faster**.
  (Caveat below.)

### ✅ Decode CUDA graph + MTP together — COLOCATED (TPOT ~17ms)

### ✅ int32→int64 decode-kernel root cause LOCATED + patch written (patch 15)

---

## 2. CAVEATS / KNOWN LIMITATIONS (honest)

### ❌ MTP does NOT compose over disagg (colocated only)
- Enabling `--speculative-config mtp` on the disagg **decode** leg crashes the EngineCore on the first request:
  a per-group KV-alloc invariant (`moriio_connector.py`, `len(local) <= len(remote)`). Measured: decode has
  **fewer** KV groups than prefill (decode=5 vs prefill=6), and MTP's speculative blocks violate the alloc
  contract. A first reconcile attempt FAILED on-device (wrong premise). **MTP is OFF on disagg** in production.
  A correct reconcile is an open item. (The reconcile code is present in the connector but non-working —
  shipped as a documented open item, NOT a working feature.)

### ⚠️ 64K decode fault on PURE-TP (TP4/TP4) — int32 offset overflow
- Pure TP4/TP4 faults at exactly 64K (illegal-address, not OOM). Root cause: KV-page-read byte offsets
  (`idx*stride_k_seq`) computed int32 in the Gluon paged mqa-logits DECODE kernel; overflow >2^31 at ~64K.
  **DP≥2 (EP8, TP4×DP2) escapes it** (per-DP-rank workspace stays smaller). **Patch 15** (int32→int64, all
  3 kernel variants) is written but **PENDING device-test**. Block-size/kpool alignment was RULED OUT by
  measurement (bs 4/16/32 all fault at 64K identically).

### ⚠️ FULL_AND_PIECEWISE — high-concurrency accuracy unverified
- Connector warns (READ mode) the per-layer KV-read barrier can't fire inside the full graph → accuracy may
  degrade at high concurrency. Recall correct at conc1. **Verify accuracy at concurrency before production.**

### ⚠️ 512K — PENDING the int32 kernel fix
- 512K overflows even DP8's per-rank offset. Needs patch 15 device-tested.

### ⚠️ TTFT floor at long context
- Router fixed the queuing; the ~28s raw 256K prefill floor is compute-bound. TP-sharded prefill (TP4×DP2)
  lowers it (25.8s). A TP-prefill/EP-decode hybrid is the next TTFT lever (needs patch 15 for pure-TP >64K).

### Build/repro gaps
- Dockerfile NOT yet test-built. Router HEAD 5f3910f is a local post-cherry-pick sha (not push-reachable);
  the Dockerfile reproduces it from UPSTREAM_REF + DPFIX_REF.

---

## 3. RECIPES (per topology) — exact serve configs

**Common (both legs, all topologies):** `--max-model-len 270000 --max-num-batched-tokens 16384
--no-enable-prefix-caching --block-size 4 --kv-cache-dtype auto`, aiter env (VLLM_ROCM_USE_AITER=1
+_MLA/_MOE/_RMSNORM=1, _FP8BMM=0, VLLM_USE_DEEP_GEMM=0, VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=4096),
MoRIIO fabric (mlx5 RoCEv2 GID=3). Prefill `--enforce-eager`. Decode
`--compilation-config '{"cudagraph_mode":"PIECEWISE"}'` + `VLLM_USE_BREAKABLE_CUDAGRAPH=1` + `--max-num-seqs 64`.
Parsers: `--tool-call-parser glm47 --reasoning-parser glm45 --enable-auto-tool-choice`.

| Topology | Parallelism (per leg) | GPUs/leg | Router RDP | Status |
|---|---|---|---|---|
| **EP8/EP8** | `--data-parallel-size 8 --enable-expert-parallel` | 8 | 8 | ✅ 256K |
| **TP4×DP2** | `--tensor-parallel-size 4 --data-parallel-size 2` (+ router `--moriio-dp-size 2`) | 8 | 2 | ✅ 256K |
| **TP4/TP4** | `--tensor-parallel-size 4` | 4 | 1 | ⚠️ ~62K (64K fault, needs patch 15) |

Serve scripts: `serve/serve_disagg_ep8_{prefill,decode}.sh`, `serve/tp4/serve_disagg_tp4_{prefill,decode_nomtp}.sh`
(+ `--data-parallel-size 2` for DP2), `serve/router/serve_router.sh` (TOPO=ep8|tp4|tp4xdp2).
Router build: `serve/router/build_router.sh`.

---

## 4. THE PATCH STACK (apply order; site-packages overlay)

| Patch | Fixes | Ships |
|---|---|---|
| core/01-05 | integration: DSA indexer dispatch, topk stub, warmup, torch.compile, gdn syntax (eager serving) | ✅ |
| core/06-07 | decode CUDA graph: opaque kpool op + fused_qk_rmsnorm (the 2 tracing walls) | ✅ |
| core/11 | flydsl fp8_mqa_logits codegen (fast prefill) | ✅ |
| core/14 | **THE recall fix** — block-table granularity (kernel_block 1152 vs spec 128) | ✅ |
| moriio_kbpb_fix/ | **disagg KV-transfer recall** — kbpb per-group DSA k_cache block expansion (factor 17 EP8 / 9 TP4, self-adapting) + get_port_offset tp_size fix (DP≥2 ZMQ collision) | ✅ |
| core/15 | int32→int64 decode mqa-logits offset (>64K / 512K on pure-TP) | ⚠️ written, PENDING device-test |
| MTP reconcile (in moriio_connector.py) | per-group block-count reconcile for MTP-over-disagg | ❌ FAILED on-device — open item, do not enable |

---

## 5. IMAGE / PROVENANCE
- Base (pin by digest — `:nightly` floats): `vllm/vllm-openai-rocm@sha256:f169e8df365b5112a1e6500543e13d8161374277c22c94d8ed60817324c101e1`
  → vLLM 0.3.1.dev3+g0bfc7a15d, torch 2.12.0, ROCm 7.2.3, aiter v0.1.19, mori v1.1.0.
- Router: upstream 0fb97775 + dpfix 82dc9811 → 5f3910f (reproduced in the Dockerfile router-build stage).
- New Dockerfile: `docker/vllm_disagg_inference.glmv53flash.mi300.ubuntu.amd.Dockerfile` (image tag `glmv5.3-flash.mi300`).

## 6. HARNESS REGISTRATION (MAD conformance)
- `models.json.entry` → add to `scripts/vllm_dissag/models.json` (name `pyt_vllm_disagg_mori_glm-5.3-flash_mi300`).
- `models.yaml.block` → add `GLM-5.3-Flash-MI300` to `scripts/vllm_dissag/models.yaml` (our config differs from
  PR254's GLM-5.3-Flash block: we run decode cudagraph, MNBT 16384).
- Add `GLM-5.3-Flash-MI300` to `VALID_MODELS` in `scripts/vllm_dissag/run_xPyD_models.slurm`.
