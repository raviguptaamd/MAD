# v4 disagg perf diagnosis — the ~160s per-request stall

## Confirmed symptom
Every disagg request (any ctx, thinking on OR off) carries a ~160-180s FIXED floor.
- 2K prompt, thinking=false, 32 out tokens: 162s / 178s (repeatable), recall CORRECT.
- 50K NIAH: 197s. 100K: 478s. => ~160s FLOOR + context-scaling prefill/transfer on top.
- GPU pinned 100% during the stall; output is correct (KV eventually arrives).
- NOT reasoning (thinking off = same), NOT config, NOT the qp knobs.

## Root cause (code-confirmed): decode->prefill notify DP-rank mismatch -> deferred wait
moriio_connector.py:708 (docstring): "Both legs of a disagg pair must agree on a single
prefill DP rank, otherwise the notify lands on a rank that never handshook and the request
HANGS until VLLM_MORIIO_DEFERRED_TIMEOUT_S."
- :810-835: connector consumes router-provided `remote_dp_rank` VERBATIM; if the router does
  not pin it (and pin the matching X-data-parallel-rank dispatch), decode notifies the wrong
  prefill rank -> that rank never RDMA-writes this request's KV -> decode waits the deferred
  timeout, then a fallback recovers the KV (why output is correct but slow).
- The connector expects an **llm-d routing sidecar** (dp_rank.go: H=pickDPRank(uuid,dp_size))
  to pin BOTH legs to the same rank H. We run the STOCK vllm-router (pin 82dc9811), which
  does round-robin dispatch but does not guarantee remote_dp_rank == dispatch rank.
- Observed: decode notifies vary (rank=2, rank=3...) but may not match where prefill actually
  ran that request -> per-request deferred stall.

## Why v3 worked (~20s/50K) but v4 stalls (~197s), same connector Python + same router
The connector Python is byte-identical to v3. The stall is the new MoRI(624002c8) WRITE/notify
path being stricter about rank matching (or a new deferred-wait default), OR the new base's DP
dispatch differs. The ~160s == a deferred-timeout default somewhere in the WRITE notify path.

## Fixes to test (in order)
1. **MORIIO_READ_MODE=1** (READ instead of WRITE): different transfer path; doc says READ has a
   "returnable path" where prefill echoes back the rank it ran -> may sidestep the notify
   mismatch entirely. FASTEST test (env only).
2. **Router rank-pinning**: make the router set kv_transfer_params.remote_dp_rank == the
   X-data-parallel-rank it dispatched (the contract at :714). Needs router code, OR a
   connector-side self-derive (compute H=hash(uuid)%dp_size on both legs identically).
3. **Lower the deferred timeout** so the fallback fires fast (masks, doesn't fix).
4. Bisect MoRI 624002c8 for the WRITE-notify behavior change vs v3's MoRI.
