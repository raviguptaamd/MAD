# syntax=docker/dockerfile:1.7
# =============================================================================
# vllm_disagg_inference.glmv53flash.mi300.ubuntu.amd.Dockerfile  (image tag: glmv5.3-flash.mi300)
#   OVERLAY build (NOT from-source): FROM the pinned vLLM ROCm nightly (which already
#   ships glm5next natively) + COPY .py patch overlays + unified diffs 14/15. No vLLM/
#   aiter/mori recompile. (The `.ubuntu.amd.Dockerfile` suffix is required for MAD's
#   engine to resolve the models.json `dockerfile` prefix; the build style is overlay.)
#   GLM-5.3-Flash-FP8 (Glm5NextForConditionalGeneration; MLA + DeepSeek Sparse
#   Attention + KDA linear attention) disaggregated 1P/1D serving image for
#   AMD MI300X (192GB) / MI325X (256GB) — gfx942. Same recipe both SKUs.
#
#   STRATEGY: OVERLAY on the pinned vLLM ROCm nightly (NO vLLM/aiter/mori
#   recompile). The nightly ALREADY ships glm5next natively + the MoRIIO
#   connector + a PREBUILT aiter (module_aiter_core.so) and mori. Our GLM-5.3
#   fixes are pure-Python .py overlays + one unified diff, applied idempotently
#   over dist-packages. So this file bakes: (a) the pinned base by DIGEST,
#   (b) the proven patch set (core 01-07,11 + patch14 recall + kbpb disagg
#   overlay), (c) a from-source vllm-router (Rust) built in a throwaway stage,
#   (d) the per-topology serve scripts at /opt/serve. Result: `docker build`
#   reproduces the ad-hoc "pinned image + docker cp patches + serve" flow as
#   ONE image. Build is ~fast (patch overlay, no 30-60min source compile) +
#   the router's ~few-min cargo build.
#
#   Modeled STRUCTURALLY on ROCm/MAD PR254
#   (docker/vllm_disagg_inference.glmv53flash.ubuntu.amd.Dockerfile): its
#   ARG/pin discipline, its section-4 router-build block, and its
#   /app/versions.txt provenance pattern. But this is gfx942/OVERLAY, NOT
#   PR254's gfx950/from-source-compile — we deliberately do NOT rebuild
#   vLLM/mori/aiter (the nightly's baked artifacts are the proven ones; a
#   wholesale PR254 overlay REGRESSED our stack, see moriio_kbpb_fix/README.md).
#
#   SOURCE OF TRUTH for every pin + rationale here: glm53_recipe/STACK.md and
#   RESULTS.md. The patch scripts under patches/ mirror the real commits on
#   an OVERLAY on the pinned vLLM ROCm nightly (glm5next ships natively; no vLLM recompile).
#
#   BUILD (single command; fill in the full 64-char digest + router ref):
#     docker build -f Dockerfile \
#       --build-arg BASE_DIGEST=sha256:f169e8df365b5112a1...<full-64-char> \
#       --build-arg ROUTER_REF=<rebased-dpfix-on-upstream-sha> \
#       -t rocmshared/glm53-flash-disagg:gfx942-mi300x .
#
#   RUN: long-lived container, then exec a serve script (see DOCKERFILE_NOTES.md
#   for the per-topology one-liners). Model mounts at /models/GLM-5.3-Flash-FP8.
# =============================================================================

# =============================================================================
# STAGE 1 — vllm-router (Rust) build. Throwaway stage: compiles the router from
#   source and emits ONLY the static binary, so the Rust toolchain + cargo
#   registry never land in the final image. Modeled on PR254 section 4.
#
#   Source = the REBASED router: raviguptaamd/router "dpfix" (82dc9811) cherry-
#   picked onto upstream vllm-project/router main @ 0fb97775. Carries the 2P2D
#   KV-notify / DP-rank deferred-write fix on top of tip (prefill_dp_round_robin
#   is already upstream). VERIFIED build 2026-09-24: vllm-router 0.1.15, resulting
#   HEAD 5f3910f, all flags present (--moriio-dp-size / moriio / --vllm-pd-
#   disaggregation / --intra-node-data-parallel-size). Build encodes the recipe
#   (clone upstream + cherry-pick dpfix), not a single push-reachable sha.
#   Pinned Rust toolchain >=1.88 (router deps time/home require rustc 1.88).
# =============================================================================
# Pin the FROM by digest — the :nightly tag has since floated to a different
# image (verified on-device 2026-09-24); this exact digest is the GLM-5.3 stack,
# locally tagged glm53-pinned. Build with the digest form to actually pin.
ARG BASE_IMAGE=vllm/vllm-openai-rocm@sha256:f169e8df365b5112a1e6500543e13d8161374277c22c94d8ed60817324c101e1
ARG BASE_DIGEST=sha256:f169e8df365b5112a1e6500543e13d8161374277c22c94d8ed60817324c101e1

FROM ${BASE_IMAGE} AS router-build

# ROUTER build = a RECIPE, not a single fetchable sha. Our production router is
# the user's dpfix commit (2P2D KV-notify: moriio_dp_size + effective_dp_size +
# remote_dp_rank_override) cherry-picked onto the LATEST upstream main. It is not
# push-reachable as one sha, so we reproduce it: clone upstream @ UPSTREAM_REF,
# cherry-pick DPFIX_REF from the fork. VERIFIED build 2026-09-24 (vllm-router
# 0.1.15): upstream 0fb97775 + cherry-pick 82dc9811 -> HEAD 5f3910f, all flags OK.
ARG UPSTREAM_REPO=https://github.com/vllm-project/router.git
ARG UPSTREAM_REF=0fb97775f219f427aff12812bdf611cb1873ccff
ARG DPFIX_REPO=https://github.com/raviguptaamd/router.git
ARG DPFIX_REF=82dc9811af17412e6e24b5942a5486bc502df23a
ARG RUST_TOOLCHAIN=1.88.0

RUN set -eu; \
    sed -i 's|http://|https://|g' /etc/apt/sources.list 2>/dev/null || true; \
    sed -i 's|http://|https://|g' /etc/apt/sources.list.d/*.list 2>/dev/null || true; \
    (apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates build-essential pkg-config libssl-dev perl make \
      && echo "LIBSSL_DEV=system" > /tmp/ssl_mode) \
    || echo "LIBSSL_DEV=vendored" > /tmp/ssl_mode; \
    rm -rf /var/lib/apt/lists/* 2>/dev/null || true

RUN set -eu; \
    if ! command -v cargo >/dev/null 2>&1; then \
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
          | sh -s -- -y --default-toolchain "${RUST_TOOLCHAIN}"; \
    fi; \
    export PATH="/root/.cargo/bin:${PATH}"; \
    rm -rf /tmp/vllm-router-src; \
    git clone --filter=blob:none "${UPSTREAM_REPO}" /tmp/vllm-router-src; \
    cd /tmp/vllm-router-src; \
    git -c advice.detachedHead=false checkout "${UPSTREAM_REF}"; \
    git remote add dpfix "${DPFIX_REPO}"; \
    git fetch --filter=blob:none dpfix "${DPFIX_REF}"; \
    git -c user.email=build@local -c user.name=build cherry-pick "${DPFIX_REF}"; \
    # Offline build env (no libssl-dev): build OpenSSL from crates.io source.
    # (Verified fix on the offline nodes; harmless when system libssl IS present.)
    if grep -q vendored /tmp/ssl_mode; then \
        cargo add openssl --features vendored 2>/dev/null \
        || printf '\n[dependencies]\nopenssl = { version = "0.10", features = ["vendored"] }\n' >> Cargo.toml; \
    fi; \
    cargo build --release; \
    install -m 755 target/release/vllm-router /usr/local/bin/vllm-router; \
    # verify the moriio disagg path + the dpfix flag are compiled in (fail if not)
    vllm-router --help 2>&1 | grep -q moriio; \
    vllm-router --help 2>&1 | grep -q moriio-dp-size; \
    mkdir -p /out && cp /usr/local/bin/vllm-router /out/vllm-router; \
    echo "VLLM_ROUTER=upstream:${UPSTREAM_REF}+dpfix:${DPFIX_REF}@$(git rev-parse HEAD) ssl=$(cat /tmp/ssl_mode)" > /out/router.provenance; \
    rm -rf /tmp/vllm-router-src /root/.cargo /root/.rustup

# =============================================================================
# STAGE 2 — final serving image. FROM the pinned base by DIGEST, overlay the
#   patch set, drop in the router binary from stage 1, bake the serve scripts.
# =============================================================================
# The nightly tag DRIFTS (different digest per node/date). PIN by digest.
# Known-good: f169e8df365b5112a1 (2026-09-17, node 015):
#   vLLM 0.3.1.dev3+g0bfc7a15d, torch 2.12.0+git6bbd260, ROCm 7.2.3,
#   aiter v0.1.19, mori v1.1.0. Only the 12-char prefix is recorded in the
#   recipe; capture the full 64-char digest from a live node
#   (`docker inspect --format '{{index .RepoDigests 0}}' \
#     vllm/vllm-openai-rocm:nightly`) and pass it as --build-arg BASE_DIGEST.
# NOTE: BASE_DIGEST is declared for provenance/labels; to actually pin the FROM
#   by digest, build with --build-arg BASE_IMAGE=vllm/vllm-openai-rocm@<digest>.
FROM ${BASE_IMAGE} AS final

# Re-declare the ARGs consumed in this stage (ARGs do not cross FROM).
ARG BASE_IMAGE
ARG BASE_DIGEST
ARG ROUTER_REPO
ARG ROUTER_REF

LABEL org.opencontainers.image.title="glm53-flash-disagg" \
      org.opencontainers.image.description="GLM-5.3-Flash-FP8 gfx942 (MI300X/MI325X) disagg — overlay on vLLM ROCm nightly" \
      glm53.base="vllm/vllm-openai-rocm:nightly @ f169e8df (0.3.1.dev3+g0bfc7a15d, torch 2.12.0, ROCm 7.2.3, aiter v0.1.19, mori v1.1.0)" \
      glm53.vllm="overlay-on-nightly@sha256:f169e8df (no fork compile)" \
      glm53.router="raviguptaamd/router dpfix rebased on vllm-project/router main 0fb97775"

ENTRYPOINT []
WORKDIR /app

# vLLM install location inside the base (all patch scripts hardcode this path).
ARG VLLM_DIST=/usr/local/lib/python3.12/dist-packages/vllm
ENV GLM53_VLLM_DIST=${VLLM_DIST}

# -----------------------------------------------------------------------------
# 1. OVERLAY PAYLOAD — the proven patch tree + apply scripts. patches/ is the
#    single source of truth; every core .py self-skips if already applied
#    (idempotent, anchor-based), patch14 is a unified diff, and the kbpb overlay
#    is a file copy over the native MoRIIO connector. Mirrors, in order:
#      serve/tp4/apply_tp4_disagg.sh  (the FULL disagg apply flow).
# -----------------------------------------------------------------------------
COPY patches/ /opt/glm53/patches/
COPY apply_patches.sh /opt/glm53/apply_patches.sh

# -----------------------------------------------------------------------------
# 2. CORE patches 01-07 + 11 — make EAGER serve correct + unlock decode CUDA
#    graph + the fast flydsl prefill path. ALL topologies (colocated + disagg)
#    need these. Rationale per patch (STACK.md section 2):
#      01-05 integration  : eager serving (ROCm dispatch alias, topk-ready stub,
#                           tilelang warmup guard, @support_torch_compile, gdn
#                           hasattr hoist)
#      06 kpool_custom_op : register kpool indexer as an opaque torch.ops.vllm op
#                           so Dynamo never traces its @triton.jit kernels ->
#                           PIECEWISE decode cudagraph captures (1st tracing wall)
#      07 fused_qk_rmsnorm: same opaque-op treatment for the 2nd tracing wall
#      11 flydsl_shrui_fix: fast prefill mqa-logits (flydsl codegen)
#    (apply_patches.sh runs exactly 01-07,11; provisional 09/10/12 are NOT
#     applied — 09 is superseded by 11, 10/12 proven irrelevant to recall.)
# -----------------------------------------------------------------------------
RUN set -eu; cd /opt/glm53; \
    bash apply_patches.sh /opt/glm53/patches; \
    echo "core 01-07,11 applied"

# -----------------------------------------------------------------------------
# 3. PATCH 14 — kpool_slot_mapping. THE recall fix (block-table granularity).
#    Unified diff over vllm/v1/attention/backends/mla/indexer.py: expands each
#    coarse shared hybrid-KV block into its `factor` fine indexer pages so every
#    DSA pool gets a unique monotonic slot (was collapsing ~7 pools onto one
#    physical slot -> nondeterministic long-context recall). Fixes BOTH the
#    write (compressed_slot_mapping) and the read (prefill chunk block_table).
#    Applied with `patch -p1 --forward` from dist-packages (the diff's a/vllm/..
#    paths); guarded by the GLM53_KPOOL_SLOT_MAPPING_FIX marker so a rebuild /
#    a base that already carries it is a no-op. Mirrors apply_tp4_disagg.sh.
# -----------------------------------------------------------------------------
RUN set -eu; \
    if grep -q GLM53_KPOOL_SLOT_MAPPING_FIX \
        "${VLLM_DIST}/v1/attention/backends/mla/indexer.py" 2>/dev/null; then \
      echo "patch14 slotfix already present — skip"; \
    else \
      cd /usr/local/lib/python3.12/dist-packages && \
      patch -p1 --forward < /opt/glm53/patches/core/14_kpool_slot_mapping.patch && \
      echo "patch14 slotfix APPLIED"; \
    fi

# -----------------------------------------------------------------------------
# 4. DISAGG-ONLY — kbpb MoRIIO overlay. KV-transfer recall fix for 1P/1D over
#    MoRIIO: the DSA indexer.k_cache is paged at a FINER kernel block than the
#    shared group block (kbpb=17 on EP8, self-adapts to 9 on TP4), so each group
#    block-id must be expanded into its kbpb kernel sub-blocks BEFORE byte
#    offsets — else every block past #0 read the wrong offset -> recall
#    corruption at the ~4096-token wall. File-copy overlay over the native
#    connector dir (layout+connector carry the fix; common+engine copied for a
#    complete matched set), then scrub pycache. Harmless for COLOCATED (the
#    connector only loads when --kv-transfer-config selects MoRIIO). See
#    patches/moriio_kbpb_fix/README.md.
# -----------------------------------------------------------------------------
RUN set -eu; \
    MORIIO="${VLLM_DIST}/distributed/kv_transfer/kv_connector/v1/moriio"; \
    if [ -d "$MORIIO" ]; then \
      cp /opt/glm53/patches/moriio_kbpb_fix/moriio_layout.py    "$MORIIO/moriio_layout.py"; \
      cp /opt/glm53/patches/moriio_kbpb_fix/moriio_connector.py "$MORIIO/moriio_connector.py"; \
      cp /opt/glm53/patches/moriio_kbpb_fix/moriio_common.py    "$MORIIO/moriio_common.py"; \
      cp /opt/glm53/patches/moriio_kbpb_fix/moriio_engine.py    "$MORIIO/moriio_engine.py"; \
      rm -rf "$MORIIO/__pycache__"; \
      echo "kbpb MoRIIO overlay applied + pycache cleared"; \
    else \
      echo "ERROR: MoRIIO connector dir not found at $MORIIO" >&2; \
      find "${VLLM_DIST}" -path '*kv_connector/v1/moriio' -type d 2>/dev/null | head; \
      exit 1; \
    fi

# -----------------------------------------------------------------------------
# 5. VERIFY the overlay took (markers + py_compile). Non-fatal: reports gaps but
#    does not fail the build, so a base that already carries some fix still
#    builds. Mirrors the VERIFY block in apply_tp4_disagg.sh.
# -----------------------------------------------------------------------------
RUN set -eu; \
    MORIIO="${VLLM_DIST}/distributed/kv_transfer/kv_connector/v1/moriio"; \
    echo "slotfix  = $(grep -c GLM53_KPOOL_SLOT_MAPPING_FIX ${VLLM_DIST}/v1/attention/backends/mla/indexer.py 2>/dev/null || echo 0)"; \
    echo "dispatch = $(grep -c GLM53_AMD_INDEXER_DISPATCH ${VLLM_DIST}/models/glm5next/amd/sparse_indexer.py 2>/dev/null || echo 0)"; \
    echo "kpool06  = $(grep -c GLM53_KPOOL_CUSTOM_OP ${VLLM_DIST}/models/glm5next/amd/sparse_indexer.py 2>/dev/null || echo 0)"; \
    echo "kbpb     = $(grep -c GLM53_INDEXER_KBPB $MORIIO/moriio_layout.py 2>/dev/null || echo 0)"; \
    python3 -c "import py_compile,sys; [py_compile.compile(f, doraise=True) for f in ['$MORIIO/moriio_layout.py','$MORIIO/moriio_connector.py','$MORIIO/moriio_common.py','$MORIIO/moriio_engine.py']]; print('moriio overlay compiles OK')" \
      || echo "verify: overlay compile reported issues (inspect above)"

# -----------------------------------------------------------------------------
# 6. vllm-router — drop in the static binary built in stage 1. No external
#    router binary needed at runtime; the toy proxy in serve/ remains available
#    for bring-up, but the real prefill-aware router is the production TTFT lever
#    (RESULTS.md: TTFT/prefill-queueing is the bottleneck at high concurrency).
# -----------------------------------------------------------------------------
COPY --from=router-build /out/vllm-router /usr/local/bin/vllm-router
COPY --from=router-build /out/router.provenance /opt/glm53/router.provenance

# -----------------------------------------------------------------------------
# 7. SERVE SCRIPTS — bake the per-topology launchers at /opt/serve. These carry
#    the CONFIG-LEVEL fixes (serve flags + env), which are deliberately NOT image
#    layers (STACK.md section 3): VLLM_USE_BREAKABLE_CUDAGRAPH=1, PIECEWISE
#    cudagraph_mode, --max-num-batched-tokens, --no-enable-prefix-caching, MTP
#    via --speculative-config, the AITER env block, and the mlx5 MoRIIO fabric
#    env. Kept in scripts so the same image serves every topology/cluster with
#    no rebuild. Includes tp4/ subtree (TP4 disagg + its self-adapting kbpb).
# -----------------------------------------------------------------------------
COPY serve/ /opt/serve/

# -----------------------------------------------------------------------------
# 8. JIT CACHE locations (structural: WHERE caches live; the launcher mounts a
#    host dir at /opt/vllm_cache to persist them across restarts).
#    CRITICAL: do NOT set AITER_JIT_DIR. The base ships a PREBUILT
#    module_aiter_core.so; pointing aiter at an empty dir forces a rebuild that
#    FAILS (module_aiter_core not found). Persist only triton/vllm caches — the
#    serve scripts set TRITON_CACHE_DIR/VLLM_CACHE_ROOT under /opt/vllm_cache.
# -----------------------------------------------------------------------------
ENV VLLM_CACHE_ROOT=/opt/vllm_cache/vllm \
    TRITON_CACHE_DIR=/opt/vllm_cache/triton_cache

# -----------------------------------------------------------------------------
# 9. PROVENANCE — one versions.txt line capturing base digest + patch set +
#    router ref, so a built image is self-describing (PR254 pattern). Config
#    knobs live in the serve scripts (section 7), not here.
# -----------------------------------------------------------------------------
RUN set -eu; \
    { \
      echo "GLM53_FLASH_DISAGG gfx942 (MI300X/MI325X) image provenance"; \
      echo "BASE=${BASE_IMAGE} @ ${BASE_DIGEST}"; \
      echo "BASE_STACK=vLLM 0.3.1.dev3+g0bfc7a15d | torch 2.12.0 | ROCm 7.2.3 | aiter v0.1.19 | mori v1.1.0"; \
      echo "PATCHES_CORE=01_amd_indexer_dispatch,02_rocm_topk_ready,03_tilelang_warmup,04_torch_compile,05_gdn_hasattr,06_kpool_custom_op,07_fused_qk_rmsnorm_op,11_flydsl_shrui_fix"; \
      echo "PATCH_RECALL=14_kpool_slot_mapping (GLM53_KPOOL_SLOT_MAPPING_FIX)"; \
      echo "DISAGG_OVERLAY=moriio_kbpb_fix (layout+connector+common+engine, GLM53_INDEXER_KBPB)"; \
      cat /opt/glm53/router.provenance 2>/dev/null || echo "VLLM_ROUTER_REF=UNKNOWN"; \
      echo "VLLM=overlay-on-nightly@sha256:f169e8df (no fork compile; glm5next native)"; \
    } > /app/versions.txt; \
    cat /app/versions.txt

# -----------------------------------------------------------------------------
# Long-lived container pattern (per STACK.md section 6):
#   docker run -d --name nite --network host --ipc host --privileged \
#     --group-add video --device /dev/kfd --device /dev/dri --cap-add IPC_LOCK \
#     --shm-size 128G --ulimit memlock=-1:-1 \
#     -v /models/GLM-5.3-Flash-FP8:/models/GLM-5.3-Flash-FP8:ro \
#     -v $HOST_CACHE:/opt/vllm_cache \
#     --entrypoint bash <image> -lc "sleep infinity"
# then `docker exec nite bash /opt/serve/<topology-script>.sh`.
# Patches are ALREADY baked — no docker cp / apply step at runtime.
# -----------------------------------------------------------------------------
WORKDIR /opt/serve
