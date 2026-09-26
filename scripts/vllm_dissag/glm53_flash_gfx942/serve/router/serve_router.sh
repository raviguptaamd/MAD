#!/bin/bash
# GLM-5.3-Flash-FP8 disagg PRODUCTION router (real Rust vllm-router).
# Runs INSIDE nite on the PREFILL/proxy node. Replaces moriio_toy_proxy_server.py
# in ALL topologies (user directive: vllm-router is production everywhere).
#
# VERIFIED 2026-09-24 on live EP8/EP8: 256K NIAH recall intact through the router,
# TTFT 4.1x better @256K conc8 vs toy proxy (196->48s P90), no collapse @conc16.
#
# The ONE knob that changes per topology is RDP (--intra-node-data-parallel-size):
#   EP8/EP8      -> RDP=8   (DP8 = 8 data-parallel ranks per leg)
#   TP4/TP4      -> RDP=1   (pure tensor-parallel, 1 DP rank per leg)
#   TP4xDP2      -> RDP=2   (2 DP ranks per leg) + set MORIIO_DP_SIZE for cross-pod
# WRONG RDP -> HTTP 400 on every request. Match it to the legs' data-parallel-size.
set -u

TOPO="${TOPO:-ep8}"                       # ep8 | tp4 | tp4xdp2
case "$TOPO" in
  ep8)     RDP="${RDP:-8}" ;;
  tp4)     RDP="${RDP:-1}" ;;
  tp4xdp2) RDP="${RDP:-2}" ;;
  *) echo "unknown TOPO=$TOPO (use ep8|tp4|tp4xdp2)"; exit 2 ;;
esac

PORT="${PORT:-10001}"                      # client-facing HTTP
DISCO="${DISCO:-0.0.0.0:36367}"            # ZMQ service discovery; legs' proxy_ping_port must match
ROUTER_BIN="${ROUTER_BIN:-/usr/local/bin/vllm-router}"
MAXCONC="${MAXCONC:-1024}"
REQTIMEOUT="${REQTIMEOUT:-3600}"
LOG="${LOG:-/opt/vllm_cache/vllm_router.log}"
# TP4xDP2 / Wide-EP cross-pod DP world size (0 => fall back to intra-node RDP).
# Load-bearing for DP>=2: makes decode KV-notify target the routed prefill DP rank
# verbatim (the dpfix). Set = total DP world when spanning pods.
MORIIO_DP_SIZE="${MORIIO_DP_SIZE:-0}"

[ -x "$ROUTER_BIN" ] || { echo "router binary not found/executable at $ROUTER_BIN"; exit 3; }

pkill -f "moriio_toy_proxy_server" 2>/dev/null || true   # retire the toy proxy if running
pkill -f "vllm-router" 2>/dev/null || true; sleep 1

EXTRA=()
[ "$MORIIO_DP_SIZE" != "0" ] && EXTRA+=(--moriio-dp-size "$MORIIO_DP_SIZE")

echo "=== vllm-router PRODUCTION (TOPO=$TOPO RDP=$RDP) $(date) ===" > "$LOG"
nohup "$ROUTER_BIN" \
  --host 0.0.0.0 --port "$PORT" \
  --vllm-pd-disaggregation \
  --kv-connector moriio \
  --vllm-discovery-address "$DISCO" \
  --intra-node-data-parallel-size "$RDP" \
  --policy round_robin --prefill-policy round_robin --decode-policy round_robin \
  --max-concurrent-requests "$MAXCONC" \
  --request-timeout-secs "$REQTIMEOUT" \
  "${EXTRA[@]}" \
  --log-level info >> "$LOG" 2>&1 &
echo "router_pid=$! (TOPO=$TOPO RDP=$RDP port=$PORT disco=$DISCO dp_size=$MORIIO_DP_SIZE)"
sleep 4
echo "--- log tail ---"; tail -12 "$LOG"
echo "--- health ---"; curl -s -m3 "http://localhost:$PORT/health" -o /dev/null -w "%{http_code}\n" || echo "(no listener yet)"
