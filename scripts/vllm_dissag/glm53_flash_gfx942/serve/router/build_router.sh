#!/bin/bash
# Build the production vllm-router: user's dpfix cherry-picked onto latest upstream.
# Runs INSIDE nite (needs github + crates.io network). Toolchain/tree persist on the
# host mount so rebuilds are cheap. VERIFIED 2026-09-24: vllm-router 0.1.15, HEAD 5f3910f.
set -eu

UPSTREAM_REPO="${UPSTREAM_REPO:-https://github.com/vllm-project/router.git}"
UPSTREAM_REF="${UPSTREAM_REF:-0fb97775f219f427aff12812bdf611cb1873ccff}"
DPFIX_REPO="${DPFIX_REPO:-https://github.com/raviguptaamd/router.git}"
DPFIX_REF="${DPFIX_REF:-82dc9811af17412e6e24b5942a5486bc502df23a}"   # 2P2D KV-notify dpfix
RUST_TOOLCHAIN="${RUST_TOOLCHAIN:-1.88.0}"

export CARGO_HOME="${CARGO_HOME:-/opt/vllm_cache/cargo}"
export RUSTUP_HOME="${RUSTUP_HOME:-/opt/vllm_cache/rustup}"
SRC="${SRC:-/opt/vllm_cache/router_build}"

if ! command -v cargo >/dev/null 2>&1 && [ ! -x "$CARGO_HOME/bin/cargo" ]; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain "$RUST_TOOLCHAIN"
fi
export PATH="$CARGO_HOME/bin:$PATH"

rm -rf "$SRC"
git clone --filter=blob:none "$UPSTREAM_REPO" "$SRC"
cd "$SRC"
git -c advice.detachedHead=false checkout "$UPSTREAM_REF"
git remote add dpfix "$DPFIX_REPO"
git fetch --filter=blob:none dpfix "$DPFIX_REF"
git -c user.email=build@local -c user.name=build cherry-pick "$DPFIX_REF"

# Offline env (nodes can't reach apt for libssl-dev): build OpenSSL from crates.io.
# Harmless if system libssl-dev IS present. This is a build-manifest edit, not router source.
if ! pkg-config --exists openssl 2>/dev/null; then
  cargo add openssl --features vendored 2>/dev/null \
    || printf '\n[dependencies]\nopenssl = { version = "0.10", features = ["vendored"] }\n' >> Cargo.toml
fi

cargo build --release
install -m 755 target/release/vllm-router /usr/local/bin/vllm-router
vllm-router --help 2>&1 | grep -q moriio
vllm-router --help 2>&1 | grep -q moriio-dp-size
echo "OK: $(vllm-router --version 2>&1) built = upstream:$UPSTREAM_REF + dpfix:$DPFIX_REF -> $(git rev-parse HEAD)"
