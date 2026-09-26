#!/bin/bash
# Smoke test: ask the disagg proxy (TP4 pair) for the capital of France.
# Runs INSIDE nite on 015. Avoids inline-JSON quote-mangling through 2-hop ssh.
set -u
PORT="${1:-11001}"
python3 - "$PORT" <<'PY'
import sys, json, urllib.request
port = sys.argv[1]
body = {
  "model": "/models/GLM-5.3-Flash-FP8",
  "messages": [{"role":"user","content":"What is the capital of France? Answer in one word."}],
  "max_tokens": 32,
  "temperature": 0.0,
}
data = json.dumps(body).encode()
url = f"http://localhost:{port}/v1/chat/completions"
try:
    req = urllib.request.Request(url, data=data, headers={"Content-Type":"application/json"})
    r = urllib.request.urlopen(req, timeout=300)
    out = json.loads(r.read())
    msg = out["choices"][0]["message"]
    txt = (msg.get("content") or "").strip()
    reason = (msg.get("reasoning_content") or "").strip()
    usage = out.get("usage", {})
    print("SMOKE_OK content=%r reason_tail=%r ptok=%s ctok=%s" % (
        txt[:120], reason[-80:], usage.get("prompt_tokens"), usage.get("completion_tokens")))
except Exception as e:
    print("SMOKE_ERR %s: %s" % (type(e).__name__, e))
PY
