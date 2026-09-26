#!/usr/bin/env python3
# Async streaming perf client for disagg proxy.
# Measures per-request TTFT, e2e latency, output tokens (from usage.completion_tokens),
# TPOT, then aggregates across the concurrency batch.
# Usage: perf_sweep.py <n_sentences> <concurrency> <max_tokens> [port] [label]
import sys, json, time, asyncio, aiohttp

N_SENT = int(sys.argv[1])
CONC = int(sys.argv[2])
MAX_TOK = int(sys.argv[3])
PORT = sys.argv[4] if len(sys.argv) > 4 else "10001"
LABEL = sys.argv[5] if len(sys.argv) > 5 else ""
MODEL = "/models/GLM-5.3-Flash-FP8"
CLIENT_TIMEOUT = 900  # seconds

sentence = "The grass is green and the sky is blue. "
context = sentence * N_SENT
prompt = context + "\n\nBriefly, what color is the sky? Answer in one short sentence."

URL = f"http://localhost:{PORT}/v1/completions"

def make_body():
    return {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": MAX_TOK,
        "ignore_eos": True,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

async def worker(session, i):
    t0 = time.time()
    ttft = None
    completion_tokens = None
    prompt_tokens = None
    chunk_count = 0
    err = None
    try:
        async with session.post(URL, json=make_body()) as resp:
            if resp.status != 200:
                body = (await resp.text())[:120]
                return {"ok": False, "err": f"HTTP{resp.status}:{body}", "dt": time.time()-t0}
            async for raw in resp.content:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    d = json.loads(data)
                except Exception:
                    continue
                # token text chunk?
                choices = d.get("choices") or []
                if choices:
                    txt = choices[0].get("text")
                    if txt:
                        if ttft is None:
                            ttft = time.time() - t0
                        chunk_count += 1
                # usage (final chunk with include_usage)
                u = d.get("usage")
                if u:
                    completion_tokens = u.get("completion_tokens")
                    prompt_tokens = u.get("prompt_tokens")
        dt = time.time() - t0
        if ttft is None:
            return {"ok": False, "err": "no_tokens_streamed", "dt": dt}
        ct = completion_tokens if completion_tokens else chunk_count
        tpot = (dt - ttft) / max(1, ct - 1) if ct and ct > 1 else 0.0
        return {"ok": True, "ttft": ttft, "dt": dt, "ct": ct,
                "pt": prompt_tokens, "chunks": chunk_count, "tpot": tpot}
    except asyncio.TimeoutError:
        return {"ok": False, "err": "client_timeout", "dt": time.time()-t0}
    except Exception as e:
        return {"ok": False, "err": f"{type(e).__name__}:{str(e)[:80]}", "dt": time.time()-t0}

async def main():
    timeout = aiohttp.ClientTimeout(total=CLIENT_TIMEOUT, sock_connect=30)
    conn = aiohttp.TCPConnector(limit=0)
    t0 = time.time()
    async with aiohttp.ClientSession(timeout=timeout, connector=conn) as session:
        tasks = [asyncio.create_task(worker(session, i)) for i in range(CONC)]
        results = await asyncio.gather(*tasks)
    wall = time.time() - t0

    ok = [r for r in results if r["ok"]]
    bad = [r for r in results if not r["ok"]]
    out = {"label": LABEL, "n_sent": N_SENT, "conc": CONC, "max_tok": MAX_TOK,
           "wall": round(wall, 2), "ok": len(ok), "attempted": CONC}
    if ok:
        ttfts = sorted(r["ttft"] for r in ok)
        def pct(a, p):
            if not a: return None
            idx = min(len(a)-1, int(round((len(a)-1)*p)))
            return a[idx]
        tpots = [r["tpot"] for r in ok if r["tpot"] > 0]
        tot_ct = sum(r["ct"] for r in ok)
        out["ttft_p50"] = round(pct(ttfts, 0.50), 2)
        out["ttft_p90"] = round(pct(ttfts, 0.90), 2)
        out["ttft_min"] = round(ttfts[0], 2)
        out["ttft_max"] = round(ttfts[-1], 2)
        out["mean_tpot_ms"] = round(sum(tpots)/len(tpots)*1000, 1) if tpots else None
        out["output_tok_s"] = round(tot_ct / wall, 1)          # aggregate OUTPUT tok/s (decode)
        # INPUT (prefill) tok/s: total prompt tokens processed / wall. Reflects TTFT/prefill throughput.
        tot_pt = sum((r.get("pt") or 0) for r in ok)
        out["input_tok_s"] = round(tot_pt / wall, 1) if tot_pt else None
        # per-request prefill rate = prompt_tokens / TTFT (raw prefill speed, contention-free view)
        pref_rates = [ (r["pt"]/r["ttft"]) for r in ok if r.get("pt") and r["ttft"]>0 ]
        out["prefill_tok_s_per_req_p50"] = round(sorted(pref_rates)[len(pref_rates)//2], 1) if pref_rates else None
        out["total_tok_s"] = round(tot_ct / wall, 1)
        out["total_out_tokens"] = tot_ct
        out["prompt_tokens_sample"] = ok[0].get("pt")
        out["completion_tokens_sample"] = ok[0].get("ct")
        out["chunks_sample"] = ok[0].get("chunks")
    if bad:
        from collections import Counter
        errs = Counter(r["err"].split(":")[0] for r in bad)
        out["errors"] = dict(errs)
        out["err_detail_sample"] = bad[0]["err"]
    print("RESULT " + json.dumps(out))

asyncio.run(main())
