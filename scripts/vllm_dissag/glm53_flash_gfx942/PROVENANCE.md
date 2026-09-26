# PROVENANCE — exact pins for the GLM-5.3-Flash-Flash gfx942 disagg image

Pin everything by digest/sha. Do not float tags. All confirmed on-device
(2026-09-24).

## Base image (pin by DIGEST — the `:nightly` tag has since floated)
```
vllm/vllm-openai-rocm:nightly
  @ sha256:f169e8df365b5112a1e6500543e13d8161374277c22c94d8ed60817324c101e1
```
- Captured 2026-09-17 (node 015), locally tagged `glm53-pinned`.
- **Currency confirmed**: this digest (`f169e8df`, tagged glm53-pinned) is NEWER than
  the live `:nightly` (`1fd21abe`) — so the pin is correct + current; pin-by-digest is
  right precisely because `:nightly` floats.
- Base stack it carries:
  - **vLLM** `0.3.1.dev3+g0bfc7a15d`
  - **torch** `2.12.0+git6bbd260`
  - **ROCm** `7.2.3`
  - **aiter** `v0.1.19` (PREBUILT `module_aiter_core.so` — kept as-is, no recompile)
  - **mori** `v1.1.0` (PREBUILT — kept as-is, no recompile)
- The base already ships `glm5next` natively + the MoRIIO connector.
- To actually digest-pin the FROM, build with
  `--build-arg BASE_IMAGE=vllm/vllm-openai-rocm@sha256:f169e8df…`.
  (`BASE_DIGEST` alone is kept for the label/versions.txt provenance line; a
  Dockerfile `FROM ${BASE_IMAGE}` cannot portably interpolate a separate digest ARG.)

## vllm-router (built from source in Dockerfile stage 1)

The production router is a **recipe, not a single push-reachable sha**: the user's
dpfix commit cherry-picked onto the latest upstream main.

```
upstream  vllm-project/router  @ 0fb97775f219f427aff12812bdf611cb1873ccff   (main)
dpfix     raviguptaamd/router  @ 82dc9811af17412e6e24b5942a5486bc502df23a   (2P2D KV-notify)
resulting HEAD                 = 5f3910f463f6b2e8a9e19c5d70cc99236a07eb66
```
- **VERIFIED build 2026-09-24**: `vllm-router 0.1.15`, 42MB static binary. Cherry-pick
  was CLEAN (58 ins / 7 del across 5 files, no manual resolution). All 4 flags present
  via `--help`: `--moriio-dp-size` (`[default:0]`, "cross-pod MoRI-IO DP world size …
  0 → fall back to intra_node_data_parallel_size"), `--kv-connector moriio`,
  `--vllm-pd-disaggregation`, `--intra-node-data-parallel-size`.
- The dpfix carries the DP-rank KV-notify override (`moriio_dp_size` +
  `effective_dp_size` + `remote_dp_rank_override`) load-bearing for TP4×DP2.
- **Offline-build note (baked into the Dockerfile)**: the build nodes are apt-offline
  (only github/crates.io reachable), so `openssl-sys` can't find system `libssl-dev`.
  Fix = vendored OpenSSL (`openssl = { version="0.10", features=["vendored"] }` — built
  from crates.io source; router source UNTOUCHED). The Dockerfile stage-1 auto-detects:
  apt-install `libssl-dev` if online, else apply the vendored-openssl Cargo edit.
- **Rust toolchain** ≥ `1.88.0` (router deps `time`/`home` require rustc 1.88).
- ⚠️ **Coordinator to confirm**: HEAD `5f3910f` is the *resulting* local sha after
  cherry-pick + the vendored-openssl working-tree edit; it is NOT push-reachable from a
  public repo. The reproducible build is the RECIPE (clone upstream `0fb97775` →
  cherry-pick raviguptaamd `82dc9811` → vendored openssl), which the Dockerfile encodes
  via `UPSTREAM_REF` + `DPFIX_REF` ARGs. Confirm those two refs before build.

## vLLM patch overlays (baked, no recompile)
- **core** `01-07,11` — eager serve + decode CUDA graph (opaque-op) + flydsl prefill.
- **patch14** `14_kpool_slot_mapping.patch` — recall fix (`GLM53_KPOOL_SLOT_MAPPING_FIX`).
- **patch15** `15_decode_mqa_logits_int64.patch` — int32→int64 decode offset (>64K/512K pure-TP; PENDING device-test).
- **moriio_kbpb_fix** `{layout,connector,common,engine}.py` — disagg KV-transfer fix
  (`GLM53_INDEXER_KBPB`).
- **vLLM provenance (accurate):** this recipe is an **OVERLAY on the pinned vLLM ROCm nightly**
  (`vllm/vllm-openai-rocm@sha256:f169e8df…`, vllm `0.3.1.dev3+g0bfc7a15d`) — **NO vLLM/aiter/mori
  recompile**. The nightly already ships `glm5next` (Glm5NextForConditionalGeneration) natively +
  the MoRIIO connector + prebuilt aiter/mori. Our only vLLM-source changes are the `patches/` `.py`
  overlays + unified diffs 14/15, applied over dist-packages. There is NO compiled vLLM fork branch
  for this recipe (unlike PR254's gfx950 `.ubuntu` build, which compiles
  `raviguptaamd/vllm@glm53-flash-disagg-upstream`). If a fork-of-record is later desired, push a real
  branch carrying these patches and cite its sha then — do not cite one that does not exist.

## Arch / hardware
- **gfx942** — MI300X (192GB) AND MI325X (256GB), identical recipe.
- Fabric: mlx5 RoCEv2, `MORI_IB_GID_INDEX=3`, `MORI_IO_DISABLE_ATOMIC_MR=1`.

## Build args to fill before `docker build`
| ARG | value | note |
|---|---|---|
| `BASE_IMAGE` | `vllm/vllm-openai-rocm@sha256:f169e8df365b5112a1e6500543e13d8161374277c22c94d8ed60817324c101e1` | full 64-char digest above |
| `BASE_DIGEST` | `sha256:f169e8df365b5112a1e6500543e13d8161374277c22c94d8ed60817324c101e1` | provenance label |
| `UPSTREAM_REF` | `0fb97775f219f427aff12812bdf611cb1873ccff` | router upstream base |
| `DPFIX_REF` | `82dc9811af17412e6e24b5942a5486bc502df23a` | router dpfix cherry-pick |
| `RUST_TOOLCHAIN` | `1.88.0` | |

## KNOWN GAPS (coordinator to confirm before submit)
1. **The Dockerfile has NOT been test-built.** It is assembled from the verified
   on-device "pinned image + docker cp patches + serve" flow. Build it once on a node
   with the base image + github/crates.io access before relying on it.
2. Router HEAD `5f3910f` is a local post-cherry-pick sha (not push-reachable); the
   Dockerfile reproduces it from `UPSTREAM_REF` + `DPFIX_REF` instead.
