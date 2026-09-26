#!/usr/bin/env python3
# Concurrency perf over disagg proxy. Streams to measure TTFT + TPOT.
# Usage: perf_disagg.py <n_tokens> <concurrency> [port]
import sys, json, time, threading, urllib.request

N = int(sys.argv[1]) if len(sys.argv) > 1 else 64000
CONC = int(sys.argv[2]) if len(sys.argv) > 2 else 8
PORT = sys.argv[3] if len(sys.argv) > 3 else "10001"

sentence = "The grass is green and the sky is blue. "
n_sent = max(1, N // 9)
context = sentence * n_sent
prompt = context + "\n\nBriefly, what color is the sky? Answer in one short sentence."

def make_body(stream):
    return json.dumps({
        "model": "/models/GLM-5.3-Flash-FP8",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 64, "temperature": 0.0, "stream": stream,
    }).encode()

results = []
def worker(i):
    t0 = time.time()
    ttft = None; ntok = 0
    try:
        req = urllib.request.Request(f"http://localhost:{PORT}/v1/chat/completions",
            data=make_body(True), headers={"Content-Type": "application/json"})
        r = urllib.request.urlopen(req, timeout=1200)
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:"): continue
            data = line[5:].strip()
            if data == "[DONE]": break
            try:
                d = json.loads(data)
                delta = d["choices"][0].get("delta", {})
                if delta.get("content") or delta.get("reasoning_content"):
                    if ttft is None: ttft = time.time() - t0
                    ntok += 1
            except Exception: pass
        dt = time.time() - t0
        tpot = (dt - ttft) / max(1, ntok - 1) if ttft and ntok > 1 else 0
        results.append((ttft, dt, ntok, tpot))
    except Exception as e:
        results.append((None, time.time() - t0, 0, str(e)[:50]))

t0 = time.time()
threads = [threading.Thread(target=worker, args=(i,)) for i in range(CONC)]
for t in threads: t.start()
for t in threads: t.join()
wall = time.time() - t0

ok = [r for r in results if r[0] is not None]
if ok:
    ttfts = sorted(r[0] for r in ok)
    tpots = sorted(r[3] for r in ok if isinstance(r[3], float) and r[3] > 0)
    tot_tok = sum(r[2] for r in ok)
    print(f"N={N} conc={CONC} ok={len(ok)}/{CONC} wall={wall:.1f}s")
    print(f"  TTFT med={ttfts[len(ttfts)//2]:.2f}s p90={ttfts[int(len(ttfts)*0.9)]:.2f}s")
    if tpots:
        print(f"  TPOT med={tpots[len(tpots)//2]*1000:.1f}ms ({1/tpots[len(tpots)//2]:.1f} tok/s/req)")
    print(f"  gen_throughput={tot_tok/wall:.1f} tok/s across {len(ok)} reqs")
else:
    print(f"N={N} conc={CONC} ALL FAILED: {results[:2]}")
