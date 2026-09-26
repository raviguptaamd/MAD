# syntax=docker/dockerfile:1.7
# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
# =============================================================================
# Dockerfile.modern — GLM-5.3-Flash-FP8 gfx942 (MI325X) FROM-SOURCE modern stack
#
#   Glm5NextForConditionalGeneration (MLA + DeepSeek Sparse Attention + KDA
#   linear attention + MTP, 288 experts, index_kpool=4) disaggregated 1P/1D
#   serving image for AMD MI325X (256GB) — gfx942. Also runs MI300X (192GB).
#
#   THIS FILE vs glm53_recipe/Dockerfile (the PROVEN OVERLAY):
#     - Dockerfile (overlay): FROM vllm/vllm-openai-rocm:nightly @f169e8df, keep
#       the nightly's PREBUILT vLLM 0.3.1.dev3 + aiter v0.1.19 + mori v1.1.0,
#       apply our .py patches on top. Fast (~mins), LOW risk, VALIDATED on-device
#       (recall to 256K EP8/EP8). THIS IS STILL THE PRODUCTION STACK.
#     - Dockerfile.modern (THIS FILE): FROM rocm/vllm-dev:ci_base (ROCm 7.2.3 /
#       torch 2.12), then COMPILE FROM SOURCE: MoRI tip + aiter fork tip +
#       glm5next-native vLLM fork branch, then apply our patches. Slow (~1-2 hr),
#       HIGHER risk (fresh compiles, ABI + anchor drift), NOT yet validated.
#       This is the "upgrade our stack to latest sources" experiment.
#
#   Modeled STRUCTURALLY on the PROVEN from-source templates:
#     refs/MAD/docker/vllm_disagg_inference.glmv5.1.ubuntu.amd.Dockerfile   (GLM-5.1)
#     refs/MAD/docker/vllm_disagg_inference.glmv53flash.ubuntu.amd.Dockerfile (PR254 GLM-5.3, gfx950)
#   Same build structure (MoRI stage -> AITER stage -> vLLM stage -> router ->
#   provenance), same flags (BUILD_UMBP=OFF, MORI_GPU_ARCHS, meson==0.64.0,
#   rustup 1.88, dmabuf/atomic-MR handled at launch), but with OUR gfx942
#   components + OUR GLM-5.3 patch overlay on top of the compiled vLLM.
#
#   *** THE vLLM DECISION (see DOCKERFILE_MODERN_NOTES.md for the full argument):
#   We COMPILE glm5next-native vLLM from raviguptaamd/vllm @ 41644d2
#   (branch glm53-flash-disagg-upstream, based on UPSTREAM vllm-project/vllm
#   @3192898754 which already ships Glm5Next + the MoRIIO connector). We do NOT:
#     - use the GLM-5.1 wideEP branch (that is GlmMoeDsaForCausalLM = GLM-5.1,
#       the WRONG model), nor
#     - rely on the ci_base's bundled vLLM (ci_base is a bare CI base; whether it
#       ships Glm5Next is NOT guaranteed — unlike the release nightly our overlay
#       proved). From-source means we pin a KNOWN glm5next vLLM commit.
#   Then we apply OUR GLM-5.3 patch overlay (core 01-07,11 + 14 recall + 15/16
#   int64 + kbpb disagg) on top of the compiled tree.
#
#   BUILD (fill the ci_base dated tag; run on the MI325 build node):
#     docker build --ulimit nofile=1048576:1048576 -f Dockerfile.modern \
#       --build-arg BASE_IMAGE=rocm/vllm-dev:ci_base-build-01a0d6a8-b743-4bb5-802c-af6a7d97bf88 \
#       -t rocmshared/glm53-flash-disagg:gfx942-mi325-modern .
#
#   RUN: long-lived container, then exec a serve script from /opt/serve. Model
#   mounts at /models/GLM-5.3-Flash-FP8. MI325 fabric env (rdma0-7 / eno0) is set
#   in the serve scripts, NOT baked — see DOCKERFILE_MODERN_NOTES.md.
# =============================================================================

# Pin to a SPECIFIC dated ci_base tag (ROCm 7.2.3 / torch 2.12), NOT floating
# :nightly. Override with the current dated tag from the MI325 build node.
ARG BASE_IMAGE=rocm/vllm-dev:ci_base-build-01a0d6a8-b743-4bb5-802c-af6a7d97bf88
FROM ${BASE_IMAGE}

ENTRYPOINT []
WORKDIR /app

# Re-declare BASE_IMAGE AFTER FROM so it is in scope for later RUN stages (the
# provenance echo). ARGs declared before FROM are only visible to FROM itself.
ARG BASE_IMAGE=rocm/vllm-dev:ci_base-build-01a0d6a8-b743-4bb5-802c-af6a7d97bf88

# gfx942 target for MI325X (AND MI300X). index_kpool=4 needs gqa64 fp8 decode; the
# aiter fork below carries the gfx942 gqa64 fix (#4957). Do NOT build gfx950 here.
ARG GFX_COMPILATION_ARCH="gfx942"
ARG PYTORCH_ROCM_ARCH="gfx942"
ARG MAX_JOBS=32

# vLLM install path inside this base — all our .py patch scripts hardcode the
# python3.12 dist-packages path. VERIFY on the ci_base (python3 --version); if the
# ci_base ships a different python minor, override GLM53_VLLM_DIST and the patches'
# hardcoded paths, or the overlay stages will SKIP (marker-guarded, non-fatal).
ARG VLLM_DIST=/usr/local/lib/python3.12/dist-packages/vllm
ENV GLM53_VLLM_DIST=${VLLM_DIST}

# -----------------------------------------------------------------------------
# 1. MoRI — ROCm/mori TIP-OF-TREE from source, gfx942, BUILD_UMBP=OFF, KEEP NIC
#    backends (do NOT pass USE_IONIC=OFF / USE_BNXT=OFF: the GLM-5.1 recipe proved
#    that deadlocks the cross-node EP all-to-all init). MoRI is JIT-built, so this
#    swaps the JIT sources the EP dispatch/combine kernels compile from at runtime.
#    UMBP is disabled (it needs gRPC not in this base; unrelated to EP kernels).
#
#    PIN: default = ROCm/mori tip 78b7a5f3 (task directive "latest"). The
#    GLM-DSA-validated fallback is 624002c897a3 (what GLM-5.1 wideEP shipped) —
#    pass --build-arg MORI_REF=624002c897a3 if tip regresses. NOTE (RISK, see
#    NOTES): PR254's GLM-5.3 path used a mori FORK (raviguptaamd/mori
#    glm53-ionic-disagg) carrying an ionic atomic-MR strip + a HIP-device-restore
#    fix. On our mlx5/rdma fabric the atomic-MR strip is done at LAUNCH via
#    MORI_IO_DISABLE_ATOMIC_MR (set in the serve scripts), but the HIP-device
#    restore is SOURCE-only — if tip ROCm/mori lacks it, the DSA indexer Triton
#    load can HIP-209 fault. If that appears, switch MORI_REPO to the fork.
# -----------------------------------------------------------------------------
ARG MORI_REPO=https://github.com/ROCm/mori.git
ARG WITH_MORI_BUILD=1
ARG MORI_REF=78b7a5f311c6e2ef4e3b929b7741ba3fed0d2fcf
ENV MORI_GPU_ARCHS=gfx942
ENV BUILD_UMBP=OFF BUILD_UMBP_SPDK=OFF
# Enable the ionic atomic-MR strip by default (safe on mlx5 — only strips
# REMOTE_ATOMIC, which the KV transfer path never uses). Serve scripts also set it.
ENV MORI_IO_DISABLE_ATOMIC_MR=1
RUN sed -i 's|http://|https://|g' /etc/apt/sources.list 2>/dev/null || true && \
    sed -i 's|http://|https://|g' /etc/apt/sources.list.d/*.list 2>/dev/null || true && \
    apt-get update && apt-get install -y --no-install-recommends \
        git build-essential cmake ninja-build ccache libssl-dev pkg-config curl ca-certificates patch && \
    pip install meson==0.64.0 "pybind11[global]" tqdm prettytable && \
    mkdir -p /app && \
    if [ "${WITH_MORI_BUILD}" != "1" ]; then \
        python3 -c "import mori, mori.io, mori.ops; print('MoRI (bundled) OK at', mori.__path__[0])" && \
        echo "MORI_REF=BUNDLED (base amd_mori, WITH_MORI_BUILD=0)" >> /app/versions.txt ; \
    else \
        pip uninstall -y amd_mori amd-mori amd-mori-nightly mori 2>/dev/null || true && \
        rm -rf /tmp/mori-src && \
        git clone --recursive "${MORI_REPO}" /tmp/mori-src && \
        cd /tmp/mori-src && git checkout "${MORI_REF}" && git submodule update --init --recursive && \
        BUILD_UMBP=OFF pip install . && \
        python3 -c "import mori, mori.io, mori.ops; print('MoRI OK at', mori.__path__[0])" && \
        echo "MORI_REF=${MORI_REF}@$(git -C /tmp/mori-src rev-parse HEAD)" >> /app/versions.txt && \
        rm -rf /tmp/mori-src ; \
    fi

# -----------------------------------------------------------------------------
# 2. AITER — raviguptaamd/aiter TIP fork from source, + flydsl, stale JIT wiped.
#    The fork tip b50066a9 carries the gfx942 gqa64 fp8-decode fix (#4957): newer
#    aiter claims native gfx942 gqa64 and routes it to a v3_ps kernel that GPU-
#    faults; the fix lets gqa64 fall through to the capture-safe persistent view-
#    fold, so cudagraph decode is kept. REQUIRED for GLM-5.3 index_kpool=4 decode.
#
#    *** flydsl PIN CORRECTION: the aiter TIP setup.py pins flydsl==0.2.2 (asserts
#    Version(flydsl)==0.2.2 at build). This is NOT the 0.3.1 the GLM-5.1 recipe
#    used — that recipe was on an OLDER aiter. Pinning 0.3.1 here would trip
#    aiter's own version assertion. We install flydsl==0.2.2 to match tip aiter.
#    (If a future aiter bumps it, read setup.py FLYDSL_VERSION and match.)
# -----------------------------------------------------------------------------
ARG AITER_REPO=https://github.com/raviguptaamd/aiter.git
ARG WITH_AITER_BUILD=1
ARG AITER_REF=b50066a91e9c94bdcd1374b687d2a47a5bfc53d5
ARG FLYDSL_VERSION=0.2.2
ENV GPU_ARCHS=${GFX_COMPILATION_ARCH} \
    PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH}
RUN if [ "${WITH_AITER_BUILD}" != "1" ]; then \
        echo "AITER: using BUNDLED base aiter (WITH_AITER_BUILD=0)" && \
        python3 -c "import importlib.metadata as m; print('aiter (bundled)', m.version('amd-aiter'))" && \
        echo "AITER_REF=BUNDLED (base amd-aiter, WITH_AITER_BUILD=0)" >> /app/versions.txt ; \
    else \
        echo "Compiling AITER fork from ${AITER_REPO}@${AITER_REF} (gfx942 gqa64 #4957)" && \
        rm -rf /tmp/aiter-src && \
        git clone --recursive "${AITER_REPO}" /tmp/aiter-src && \
        cd /tmp/aiter-src && git checkout "${AITER_REF}" && \
        git submodule update --init --recursive && \
        (pip uninstall -y amd_aiter amd-aiter aiter 2>/dev/null || true) && \
        pip install --no-deps -U "flydsl==${FLYDSL_VERSION}" && \
        pip install --no-build-isolation --no-deps -v . && \
        echo "AITER_REF=${AITER_REF}@$(git rev-parse HEAD) (fork tip + #4957 gqa64; flydsl==${FLYDSL_VERSION})" >> /app/versions.txt && \
        rm -rf /tmp/aiter-src && \
        rm -rf /opt/vllm_cache/aiter_jit /root/.aiter && echo "cleared stale AITER JIT cache" ; \
    fi

# -----------------------------------------------------------------------------
# 3. vLLM — COMPILE glm5next-native from source. See the vLLM DECISION header +
#    NOTES for why this branch and not the bundled ci_base vLLM nor GLM-5.1 wideEP.
#    Branch glm53-flash-disagg-upstream @ 41644d2 is based on UPSTREAM
#    vllm-project/vllm @3192898754, which already ships Glm5Next + the MoRIIO
#    connector + SupportsHMA + the normalized MLA KV layout, PLUS 5 ROCm GLM-5.3
#    disagg fixes not yet upstream (moriio_layout AITER sparse-MLA geometry;
#    rocm_aiter_mla_sparse record_logical_topk_ready; get_port_offset TP-awareness;
#    relaxed HMA block guards; per-group KV routing + deferred-write). One image
#    serves BOTH TP4 1P/1D and EP8 1P/1D — they differ only in launch env.
#    This is a FULL source compile (~30-60 min): the ci_base ships a different
#    (or no) vLLM, so a .py-only overlay would be ABI-mismatched.
# -----------------------------------------------------------------------------
ARG VLLM_REPO=https://github.com/raviguptaamd/vllm.git
ARG WITH_VLLM_BUILD=1
# BEST-OF-BOTH vLLM: base = glm53-flash-disagg-upstream (41644d2, freshest upstream
# Sept-14 base + newest per-group KV block routing on WRITE path + hybrid-KV disagg
# boot fixes), THEN cherry-pick the 4 DSA-fault/EP32 fixes that ONLY exist on the
# sibling branch glm53-flash-moriio-mla-fix (verified: all 4 apply CLEAN cumulatively
# onto 41644d2 -> HEAD c3530558b). Without these the disagg-upstream base is MISSING
# the "fixes disagg decode GPU fault" commit — exactly our DSA-decode fault class.
ARG VLLM_REF=41644d2afad75adf5d6d8f55c1cee92b5147cb09
# 4 cherry-picks (chronological; from moriio-mla-fix line), all apply clean:
#   0344b77e2 [ROCm][DSA] Bounds-guard sparse-indexer k-cache Triton kernel
#   cda364860 [ROCm][DSA] Revert invalid-index sentinel -1->0 (fixes disagg decode GPU fault)
#   e8c186f71 [MoE][ROCm] Make MoRI EP sizing env-tunable (3.4x decode speedup)
#   623fdc946 [MoE][ROCm] MoRI combine() pre-dispatch topk_ids (fixes 4P/4D EP32)
ARG VLLM_CHERRYPICKS="0344b77e2 cda364860 e8c186f71 623fdc946"
ENV VLLM_TARGET_DEVICE=rocm \
    PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH} \
    MAX_JOBS=${MAX_JOBS}
RUN if [ "${WITH_VLLM_BUILD}" != "1" ]; then \
        echo "vLLM: using BUNDLED ci_base vLLM (WITH_VLLM_BUILD=0) — VERIFY it ships Glm5Next!" && \
        python3 -c "import vllm; print('vLLM (bundled)', vllm.__version__, vllm.__file__)" && \
        python3 -c "import vllm.models.glm5next" 2>/dev/null \
          && echo "BUNDLED vLLM HAS glm5next" \
          || echo "WARNING: BUNDLED vLLM has NO glm5next module — overlay will fail" && \
        echo "VLLM_REF=BUNDLED (ci_base vLLM, WITH_VLLM_BUILD=0)" >> /app/versions.txt ; \
    else \
        rm -rf /tmp/vllm-src && \
        git clone "${VLLM_REPO}" /tmp/vllm-src && \
        cd /tmp/vllm-src && \
        git -c advice.detachedHead=false checkout "${VLLM_REF}" && \
        for cp in ${VLLM_CHERRYPICKS}; do \
          echo "cherry-pick $cp" && \
          git -c user.email=build@local -c user.name=build cherry-pick "$cp" \
            || { echo "FATAL: cherry-pick $cp conflicted (unexpected — verified clean 2026-09-25)"; exit 1; } ; \
        done && \
        echo "VLLM_REF=${VLLM_REF}+cherrypicks[${VLLM_CHERRYPICKS}]@$(git rev-parse HEAD) (glm5next native, best-of-both)" >> /app/versions.txt && \
        pip uninstall -y vllm 2>/dev/null || true && \
        pip install --no-deps --no-build-isolation -v . && \
        python3 -c "import vllm; print('vLLM', vllm.__version__, 'from', vllm.__file__)" && \
        python3 -c "import vllm.models.glm5next; print('glm5next module present')" && \
        rm -rf /tmp/vllm-src ; \
    fi

# Cross-check MoRI + AITER survived the vLLM install (no silent downgrade).
RUN python3 - <<'PYEOF'
from importlib.metadata import version as v, PackageNotFoundError
def get(names):
    for n in names:
        try: return v(n)
        except PackageNotFoundError: pass
    return None
av = get(("amd-aiter", "amd_aiter", "aiter"))
assert av, "AITER missing after vLLM install (expected source-built fork tip)"
import mori, mori.io, mori.ops
print("Post-vLLM check OK: AITER", av, "present + MoRI importable")
PYEOF

# =============================================================================
# 4. OUR GLM-5.3 PATCH OVERLAY — applied on top of the compiled glm5next vLLM.
#    Same payload + apply flow as glm53_recipe/Dockerfile (the proven overlay),
#    modeled on serve/tp4/apply_tp4_disagg.sh (the FULL disagg apply flow).
#    Every core .py self-skips if already applied (idempotent, anchor-based);
#    14 is a unified diff (marker-guarded); kbpb is a file-copy overlay.
#
#    *** REDUNDANCY / CONFLICT RISK (see NOTES): the vLLM fork branch already
#    carries SOME of these fixes IN-SOURCE. In particular the kbpb file-COPY
#    overlay REPLACES moriio_layout.py/connector.py wholesale — if the fork's
#    in-source moriio fixes differ, the copy CLOBBERS them. The core anchor
#    patches self-skip when the anchor is absent (already fixed) so they are
#    safe; kbpb is the one to watch. This is why the kbpb stage is TOGGLEABLE.
#    DECISION (2026-09-25): default 0 — the vLLM fork branch 41644d2's TOP commit is
#    "GLM-5.3-Flash disagg: per-group KV block routing on WRITE path" = the fork
#    already does our kbpb IN-SOURCE. So DON'T clobber it; trust the fork's moriio.
#    (Device-test verifies the fork's routing matches our proven kbpb recall; if it
#    fails, rebuild with --build-arg WITH_KBPB_OVERLAY=1 to force our overlay.)
# -----------------------------------------------------------------------------
ARG WITH_KBPB_OVERLAY=0
COPY patches/ /opt/glm53/patches/
COPY apply_patches.sh /opt/glm53/apply_patches.sh

# 4a. CORE 01-07,11 — eager-serve correctness + decode CUDA graph opaque-op +
#     flydsl fast prefill. All topologies need these. Idempotent/self-skipping.
RUN set -eu; cd /opt/glm53; \
    bash apply_patches.sh /opt/glm53/patches; \
    echo "core 01-07,11 apply attempted (self-skips where fork already carries the fix)"

# 4b. PATCH 14 — kpool_slot_mapping, THE recall fix (block-table granularity).
#     patch -p1 --forward from dist-packages; marker-guarded so a fork that
#     already carries it is a no-op (verify step below shows slotfix=count).
RUN set -eu; \
    if grep -q GLM53_KPOOL_SLOT_MAPPING_FIX \
        "${VLLM_DIST}/v1/attention/backends/mla/indexer.py" 2>/dev/null; then \
      echo "patch14 slotfix already present — skip"; \
    else \
      cd "$(dirname ${VLLM_DIST})" && \
      patch -p1 --forward < /opt/glm53/patches/core/14_kpool_slot_mapping.patch \
        && echo "patch14 slotfix APPLIED" \
        || echo "patch14 slotfix did NOT apply cleanly (fork indexer.py may differ — inspect)"; \
    fi

# 4c. PATCH 15/16 — int32->int64 decode-MQA + MTP spec-path offset widening.
#     Idempotent/guarded; may be REDUNDANT if the tip aiter already widened these
#     kernels (task note). Apply forward, tolerate an already-applied no-op.
#     NOTE PATH: 15 targets aiter/ops/triton/gluon/pa_mqa_logits.py inside the
#     aiter install (NOT vllm dist) — verify the installed aiter path first.
RUN set -eu; \
    AITER_DIR="$(python3 -c 'import aiter,os;print(os.path.dirname(aiter.__file__))' 2>/dev/null || true)"; \
    if [ -n "${AITER_DIR}" ] && [ -f "${AITER_DIR}/ops/triton/gluon/pa_mqa_logits.py" ]; then \
      cd "$(dirname ${AITER_DIR})" && \
      patch -p1 --forward < /opt/glm53/patches/core/15_decode_mqa_logits_int64.patch \
        && echo "patch15 int64 decode-MQA APPLIED" \
        || echo "patch15 skipped/redundant (already int64 in tip aiter, or path differs)"; \
    else \
      echo "patch15: aiter gluon pa_mqa_logits.py not found — SKIP (verify tip aiter kernel layout)"; \
    fi; \
    cd "$(dirname ${VLLM_DIST})" && \
    patch -p1 --forward < /opt/glm53/patches/core/16_mtp_specpath_int64.patch \
      && echo "patch16 MTP spec-path int64 APPLIED" \
      || echo "patch16 skipped/redundant (already int64, or fork kernel differs)"

# 4d. DISAGG kbpb MoRIIO overlay — per-group hybrid-KV block expansion (kbpb=17
#     DSA indexer k_cache). File-copy over the connector dir. TOGGLEABLE: default
#     on; set WITH_KBPB_OVERLAY=0 to trust the fork's in-source per-group routing.
RUN set -eu; \
    if [ "${WITH_KBPB_OVERLAY}" != "1" ]; then \
      echo "kbpb overlay DISABLED (WITH_KBPB_OVERLAY=0) — trusting fork in-source moriio"; \
    else \
      MORIIO="${VLLM_DIST}/distributed/kv_transfer/kv_connector/v1/moriio"; \
      if [ -d "$MORIIO" ]; then \
        cp /opt/glm53/patches/moriio_kbpb_fix/moriio_layout.py    "$MORIIO/moriio_layout.py"; \
        cp /opt/glm53/patches/moriio_kbpb_fix/moriio_connector.py "$MORIIO/moriio_connector.py"; \
        cp /opt/glm53/patches/moriio_kbpb_fix/moriio_common.py    "$MORIIO/moriio_common.py"; \
        cp /opt/glm53/patches/moriio_kbpb_fix/moriio_engine.py    "$MORIIO/moriio_engine.py"; \
        rm -rf "$MORIIO/__pycache__"; \
        echo "kbpb MoRIIO overlay applied + pycache cleared"; \
      else \
        echo "WARNING: MoRIIO connector dir not found at $MORIIO (fork layout differs?)" >&2; \
        find "${VLLM_DIST}" -path '*kv_connector/v1/moriio' -type d 2>/dev/null | head; \
      fi; \
    fi

# 4e. VERIFY the overlay took (markers + py_compile). Non-fatal: reports gaps.
RUN set -eu; \
    MORIIO="${VLLM_DIST}/distributed/kv_transfer/kv_connector/v1/moriio"; \
    echo "slotfix  = $(grep -c GLM53_KPOOL_SLOT_MAPPING_FIX ${VLLM_DIST}/v1/attention/backends/mla/indexer.py 2>/dev/null || echo 0)"; \
    echo "dispatch = $(grep -c GLM53_AMD_INDEXER_DISPATCH ${VLLM_DIST}/models/glm5next/amd/sparse_indexer.py 2>/dev/null || echo 0)"; \
    echo "kpool06  = $(grep -c GLM53_KPOOL_CUSTOM_OP ${VLLM_DIST}/models/glm5next/amd/sparse_indexer.py 2>/dev/null || echo 0)"; \
    echo "kbpb     = $(grep -c GLM53_INDEXER_KBPB $MORIIO/moriio_layout.py 2>/dev/null || echo 0)"; \
    python3 -c "import py_compile,sys; [py_compile.compile(f, doraise=True) for f in ['$MORIIO/moriio_layout.py','$MORIIO/moriio_connector.py']]; print('moriio overlay compiles OK')" \
      2>/dev/null || echo "verify: moriio overlay compile check skipped/failed (inspect above)"

# -----------------------------------------------------------------------------
# 5. vllm-router — RECIPE (not a single sha): clone latest upstream
#    vllm-project/router @0fb97775, cherry-pick our dpfix raviguptaamd/router
#    @82dc9811 (2P2D KV-notify: moriio_dp_size + effective_dp_size +
#    remote_dp_rank_override). Built in-image; toolchain removed at the end.
#    Same recipe as serve/router/build_router.sh + glm53_recipe/Dockerfile.
#    Pinned Rust >=1.88 (router deps time/home require rustc 1.88).
# -----------------------------------------------------------------------------
ARG UPSTREAM_REPO=https://github.com/vllm-project/router.git
ARG UPSTREAM_REF=0fb97775f219f427aff12812bdf611cb1873ccff
ARG DPFIX_REPO=https://github.com/raviguptaamd/router.git
ARG DPFIX_REF=82dc9811af17412e6e24b5942a5486bc502df23a
ARG RUST_TOOLCHAIN=1.88.0
RUN set -eu; \
    if ! command -v cargo >/dev/null 2>&1; then \
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain "${RUST_TOOLCHAIN}"; \
    fi; \
    export PATH="/root/.cargo/bin:${PATH}"; \
    rm -rf /tmp/vllm-router-src; \
    git clone --filter=blob:none "${UPSTREAM_REPO}" /tmp/vllm-router-src; \
    cd /tmp/vllm-router-src; \
    git -c advice.detachedHead=false checkout "${UPSTREAM_REF}"; \
    git remote add dpfix "${DPFIX_REPO}"; \
    git fetch --filter=blob:none dpfix "${DPFIX_REF}"; \
    git -c user.email=build@local -c user.name=build cherry-pick "${DPFIX_REF}"; \
    if ! pkg-config --exists openssl 2>/dev/null; then \
        cargo add openssl --features vendored 2>/dev/null \
        || printf '\n[dependencies]\nopenssl = { version = "0.10", features = ["vendored"] }\n' >> Cargo.toml; \
    fi; \
    cargo build --release; \
    install -m 755 target/release/vllm-router /usr/local/bin/vllm-router; \
    vllm-router --help 2>&1 | grep -q moriio; \
    vllm-router --help 2>&1 | grep -q moriio-dp-size; \
    echo "VLLM_ROUTER=upstream:${UPSTREAM_REF}+dpfix:${DPFIX_REF}@$(git rev-parse HEAD)" >> /app/versions.txt; \
    rm -rf /tmp/vllm-router-src /root/.cargo /root/.rustup

# -----------------------------------------------------------------------------
# 6. SERVE SCRIPTS — bake per-topology launchers at /opt/serve. These carry the
#    CONFIG-LEVEL fixes (deliberately NOT image layers): VLLM_USE_BREAKABLE_
#    CUDAGRAPH=1, PIECEWISE cudagraph_mode, --max-num-batched-tokens, --no-enable-
#    prefix-caching, MTP --speculative-config, the AITER env block, --block-size 4
#    (GLM-5.3 index_kpool=4 REQUIRES block_size multiple of 4; block-size 1 is
#    FORBIDDEN, unlike GLM-5.1), and the MoRIIO fabric env.
#    *** MI325 FABRIC: the scripts default to the MI300X cluster's mlx5_* / eth0.
#    On the MI325 cluster the fabric is rdma0-7 / eno0 — edit RDMA_DEVICES,
#    MORI_*_IFNAME, NCCL_*_IFNAME, GLOO_SOCKET_IFNAME in the serve scripts (or
#    override via env). See DOCKERFILE_MODERN_NOTES.md.
# -----------------------------------------------------------------------------
COPY serve/ /opt/serve/

# -----------------------------------------------------------------------------
# 7. JIT CACHE locations (structural). For a FROM-SOURCE build the aiter JIT is
#    built from the fork source, so AITER_JIT_DIR CAN be persisted (unlike the
#    overlay-on-nightly, where setting it broke the prebuilt .so). We persist
#    aiter/triton/vllm/comgr caches under /opt/vllm_cache (host bind-mount).
# -----------------------------------------------------------------------------
ENV AITER_JIT_DIR=/opt/vllm_cache/aiter_jit \
    VLLM_CACHE_ROOT=/opt/vllm_cache/vllm \
    TRITON_CACHE_DIR=/opt/vllm_cache/triton \
    COMGR_CACHE_DIR=/opt/vllm_cache/comgr

# -----------------------------------------------------------------------------
# 8. CRITICAL — scrub build-time MoRI JIT state. The `import mori` checks above
#    compile/lock MoRI EP kernels under /root/.mori/jit on THIS build host,
#    leaving stale .hsaco.lock files. At runtime MoriAll2AllManager waits on a
#    build-in-progress whose owner PID is gone and DEADLOCKS at ep:0 init. Ship
#    /root/.mori empty so runtime compiles fresh.
# -----------------------------------------------------------------------------
RUN rm -rf /root/.mori /tmp/mori_jit_* && mkdir -p /root/.mori && \
    echo "JIT_SCRUBBED: /root/.mori + /tmp/mori_jit_* cleared at build end" >> /app/versions.txt

# -----------------------------------------------------------------------------
# 9. PROVENANCE — versions.txt (from-source pins + patch set + router recipe).
# -----------------------------------------------------------------------------
LABEL org.opencontainers.image.title="glm53-flash-disagg-modern" \
      org.opencontainers.image.description="GLM-5.3-Flash-FP8 gfx942 (MI325X) FROM-SOURCE modern stack: ci_base + MoRI tip + aiter fork tip + glm5next vLLM + our patches" \
      glm53.mori="ROCm/mori@78b7a5f3 (tip, gfx942, UMBP OFF)" \
      glm53.aiter="raviguptaamd/aiter@b50066a9 (tip fork, gfx942 gqa64 #4957, flydsl 0.2.2)" \
      glm53.vllm="raviguptaamd/vllm@41644d2 glm53-flash-disagg-upstream (glm5next native) + our patches" \
      glm53.router="vllm-project/router@0fb97775 + raviguptaamd/router dpfix@82dc9811"
RUN set -eu; \
    { \
      echo "GLM53_FLASH_DISAGG_MODERN gfx942 (MI325X/MI300X) FROM-SOURCE image provenance"; \
      echo "BASE=${BASE_IMAGE} (ci_base ROCm 7.2.3 / torch 2.12)"; \
      echo "PATCHES_CORE=01_amd_indexer_dispatch,02_rocm_topk_ready,03_tilelang_warmup,04_torch_compile,05_gdn_hasattr,06_kpool_custom_op,07_fused_qk_rmsnorm_op,11_flydsl_shrui_fix"; \
      echo "PATCH_RECALL=14_kpool_slot_mapping (GLM53_KPOOL_SLOT_MAPPING_FIX)"; \
      echo "PATCH_INT64=15_decode_mqa_logits_int64,16_mtp_specpath_int64 (idempotent; may be redundant w/ tip aiter)"; \
      echo "DISAGG_OVERLAY=moriio_kbpb_fix (GLM53_INDEXER_KBPB; WITH_KBPB_OVERLAY=${WITH_KBPB_OVERLAY})"; \
      echo "BLOCK_SIZE=4 (index_kpool=4 REQUIRES multiple of 4; block-size 1 FORBIDDEN on 5.3)"; \
    } >> /app/versions.txt; \
    cat /app/versions.txt

# -----------------------------------------------------------------------------
# Long-lived container pattern:
#   docker run -d --name nite --network host --ipc host --privileged \
#     --group-add video --device /dev/kfd --device /dev/dri --cap-add IPC_LOCK \
#     --shm-size 128G --ulimit memlock=-1:-1 \
#     -v /models/GLM-5.3-Flash-FP8:/models/GLM-5.3-Flash-FP8:ro \
#     -v $HOST_CACHE:/opt/vllm_cache \
#     --entrypoint bash <image> -lc "sleep infinity"
# then `docker exec nite bash /opt/serve/<topology-script>.sh` (edit MI325 fabric).
# Patches are ALREADY baked — no docker cp / apply step at runtime.
# -----------------------------------------------------------------------------
WORKDIR /opt/serve
