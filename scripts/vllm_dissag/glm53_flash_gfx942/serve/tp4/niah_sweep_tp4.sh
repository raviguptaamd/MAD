#!/bin/bash
# NIAH ceiling sweep for the TP4/TP4 disagg pair, through proxy :11001.
# depth 0.5 across ascending lengths; single rep per cell (fast ceiling find).
# Uses /opt/vllm_cache/niah_disagg.py (needle 74923, ~10 tok/sentence, max_tokens 256).
set -u
H=/opt/vllm_cache/niah_disagg.py
PORT=11001
RES=/opt/vllm_cache/NIAH_TP4.txt
LENS="${LENS:-4000 8000 16000 32000 64000 100000}"
DEPTH="${DEPTH:-0.5}"
echo "=== NIAH TP4 SWEEP $(date -u +%FT%TZ) port=$PORT depth=$DEPTH lens=[$LENS] ===" | tee -a "$RES"
for L in $LENS; do
  line=$(python3 "$H" "$L" "$DEPTH" "$PORT" 256 2>&1)
  verdict=$(echo "$line" | grep -oE 'NEEDLE=(PASS|FAIL)')
  echo "LEN=$L DEPTH=$DEPTH -> ${verdict:-ERR} | $line" | tee -a "$RES"
done
echo "=== TP4 SWEEP DONE $(date -u +%FT%TZ) ===" | tee -a "$RES"
