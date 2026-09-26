# Production router (vllm-router) — all disagg topologies

The real Rust `vllm-router` is THE production proxy for every disaggregated config
(EP8/EP8, TP4/TP4, TP4×DP2). The old `moriio_toy_proxy_server.py` is retired to
`../debug_toyproxy/` (fallback/debug only — it serializes prefill and cannot route
DP-rank KV-notify, so it wedges on DP≥2).

## Build (once, on a node with github + crates.io access)
```
docker exec nite bash /opt/serve/router/build_router.sh
```
Produces `/usr/local/bin/vllm-router` = upstream vllm-project/router @0fb97775 +
user's 2P2D KV-notify dpfix (raviguptaamd/router @82dc9811). Verified: vllm-router 0.1.15.

## Launch (on the prefill/proxy node, after both legs are up)
```
TOPO=ep8     bash /opt/serve/router/serve_router.sh     # RDP=8
TOPO=tp4     bash /opt/serve/router/serve_router.sh     # RDP=1
TOPO=tp4xdp2 MORIIO_DP_SIZE=2 bash /opt/serve/router/serve_router.sh   # RDP=2 + dpfix
```
`RDP` (`--intra-node-data-parallel-size`) MUST match the legs' `--data-parallel-size`
(8 EP8 / 1 TP4 / 2 TP4×DP2) or every request 400s. Bring order: decode leg up, then
prefill, then this router. Legs auto-register with the discovery ZMQ (`--vllm-discovery-address`,
default :36367) via their ping threads — no leg restart needed if the port matches the legs'
`proxy_ping_port`. Warm 2–3 tiny requests (mori CreateSession cold-start can 503 the first).

## Measured (EP8/EP8, vs the retired toy proxy)
256K NIAH recall intact through the router. TTFT @256K conc8: 196→48s P90 (4.1×);
conc16: 229/396→90/94s (no collapse). See ../../RESULTS.md.

## For TP4×DP2 (DP2)
The dpfix (`--moriio-dp-size`) is LOAD-BEARING: plain upstream re-hashes the KV-notify
target as `blake2s(request_id) % dp_size` ≠ the routed prefill rank → decode notifies the
wrong rank → "remote blocks never arrived" wedge. The dpfix forces both legs to honor the
routed rank verbatim. Set `MORIIO_DP_SIZE` = total cross-pod DP world size.
