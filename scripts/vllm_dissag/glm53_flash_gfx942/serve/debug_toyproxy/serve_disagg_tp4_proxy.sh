#!/bin/bash
# Runs INSIDE nite on node 015 (PREFILL/proxy node). MoRIIO toy proxy for the TP4 pair.
# HTTP proxy :11001 ; service-discovery :37367 (matches the KV configs' proxy_ping_port).
# The stock proxy hardcodes discovery :36367, so we sed a copy to :37367 (+1000 shift)
# to keep this TP4 pair fully isolated from the live EP8 pair (proxy :10001 / disc :36367).
set -u
SRC=/app/vllm/examples/disaggregated/disaggregated_serving/moriio_toy_proxy_server.py
PROXY=/opt/vllm_cache/moriio_toy_proxy_tp4.py
LOG=/opt/vllm_cache/tp4_proxy.log
# Build the shifted-discovery copy (idempotent: regenerate each launch from source).
sed 's/36367/37367/g' "$SRC" > "$PROXY"
pkill -f moriio_toy_proxy_tp4 2>/dev/null; sleep 1
echo "=== MoRIIO toy proxy :11001 (discovery :37367) $(date) ===" > $LOG
nohup python3 "$PROXY" --port 11001 >> $LOG 2>&1 &
echo "proxy_pid=$!"
sleep 4
echo "=== log tail ==="; tail -15 $LOG
echo "=== ports ==="; ss -ltnp 2>/dev/null | grep -E ":11001|:37367" || echo "(no listener yet)"
