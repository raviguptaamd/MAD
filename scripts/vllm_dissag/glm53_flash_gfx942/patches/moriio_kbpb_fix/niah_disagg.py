#!/usr/bin/env python3
# NIAH over disagg proxy. Needle 74923. Sends chat/completions to proxy :10001.
# Usage: niah_disagg.py <n_tokens> [depth_frac] [port]
import sys, json, time, urllib.request

N = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
DEPTH = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
PORT = sys.argv[3] if len(sys.argv) > 3 else "10001"
NEEDLE = "74923"

# Build filler ~N tokens. Insert needle at DEPTH.
# CALIBRATION: measured 240000 -> 266,709 real prompt tokens on this tokenizer, i.e. ~10 tok/sentence
# (the old value 9 UNDER-counted, so N over-inflated: N=256000 tokenized to 269,745 -> 1 tok over
# MML 270000 -> prefill 400. With 10, N ~= real prompt-token target, leaving margin under a 270K MML.)
sentence = "The grass is green and the sky is blue. "
approx_tok_per = 10
n_sent = max(1, N // approx_tok_per)
filler = [sentence] * n_sent
ins = int(len(filler) * DEPTH)
needle_sent = f"The special magic number is {NEEDLE}. Remember it. "
filler.insert(ins, needle_sent)
context = "".join(filler)
prompt = (context +
    "\n\nQuestion: What is the special magic number mentioned in the text above? "
    "Answer with ONLY the number, nothing else.")

# NIAH answer is a short number; cap output low so (prompt + max_tokens) stays under
# --max-model-len even at 256K. (12000 overshot MML by 1 tok at 256K -> prefill 400.)
MAXTOK = int(sys.argv[4]) if len(sys.argv) > 4 else 256
body = {
    "model": "/models/GLM-5.3-Flash-FP8",
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": MAXTOK,
    "temperature": 0.0,
}
data = json.dumps(body).encode()
url = f"http://localhost:{PORT}/v1/chat/completions"
t0 = time.time()
try:
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    r = urllib.request.urlopen(req, timeout=1200)
    out = json.loads(r.read())
    dt = time.time() - t0
    msg = out["choices"][0]["message"]
    txt = msg.get("content") or ""
    reason = msg.get("reasoning_content") or ""
    usage = out.get("usage", {})
    ptok = usage.get("prompt_tokens", "?")
    ctok = usage.get("completion_tokens", "?")
    found = NEEDLE in txt or NEEDLE in reason
    # last 200 chars of content, and whether needle in reasoning
    tail = txt.strip()[-200:]
    print(f"N={N} depth={DEPTH} ptok={ptok} ctok={ctok} dt={dt:.1f}s "
          f"NEEDLE={'PASS' if found else 'FAIL'} in_content={NEEDLE in txt} "
          f"in_reason={NEEDLE in reason}")
    print(f"  content_tail={tail!r}")
except Exception as e:
    print(f"N={N} depth={DEPTH} ERROR {type(e).__name__}: {e}")
