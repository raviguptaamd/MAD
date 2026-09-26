#!/bin/bash
# Runs INSIDE nite on 030 (PREFILL/proxy node). MoRIIO toy proxy.
# HTTP proxy :10001 ; service-discovery hardcoded :36367 (matches KV proxy_ping_port).
set -u
PROXY=/app/vllm/examples/disaggregated/disaggregated_serving/moriio_toy_proxy_server.py
LOG=/opt/vllm_cache/disagg_proxy.log
pkill -f moriio_toy_proxy_server 2>/dev/null; sleep 1
echo "=== MoRIIO toy proxy :10001 (discovery :36367) $(date) ===" > $LOG
nohup python3 $PROXY --port 10001 >> $LOG 2>&1 &
echo "proxy_pid=$!"
sleep 4
echo "=== log tail ==="; tail -15 $LOG
echo "=== ports ==="; ss -ltnp 2>/dev/null | grep -E ":10001|:36367" || echo "(no listener yet)"
