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

## FIX IN TEST: MORIIO_READ_MODE=1 (returnable path)
CONFIRMED the exact mechanism (moriio_connector.py:1190-1210 + 794/816-835):
- Router (pin 82dc9811, vllm_pd_router.rs:391-490/697-845) sets the X-data-parallel-rank
  DISPATCH header (round-robin) but does NOT set kv_transfer_params.remote_dp_rank.
- In WRITE mode, decode does remote_dp_rank = kv_transfer_params.get("remote_dp_rank", 0)
  -> DEFAULTS TO 0 -> always notifies prefill rank 0. But prefill ran on a round-robin rank
  != 0 -> notify misroutes -> decode waits the deferred timeout (~160s) then a fallback
  recovers the KV (correct output, ~160s late).
- WRITE mode is NOT "returnable": prefill's request_finished echo of remote_dp_rank only
  reaches decode on READ / serial-WRITE paths (:719-720, :1190).
FIX: MORIIO_READ_MODE=1 -> READ path IS returnable. request_finished (:1200-1210) returns
remote_dp_rank=self._global_dp_rank (the rank prefill ACTUALLY ran) + remote_dp_rank_override
=True; router forwards prefill's kv_transfer_params to decode (vllm_pd_router.rs:420-432);
decode gate _should_notify = (self._global_dp_rank == remote_dp_rank) -> notifies the CORRECT
rank -> NO deferred stall. Expected: per-request latency drops from ~160s to ~seconds.
STATUS: relaunched with MORIIO_READ_MODE=1, keeping thinking=false/320K/F_A_P. Validating.

## READ_MODE test RESULT: FAILED (rejected)
MORIIO_READ_MODE=1 + F_A_P: request still 149.2s (stall NOT fixed) AND output GARBAGE
("与 相似 相似..." repeated CJK, not "8241"). The READ+FULL_AND_PIECEWISE combo breaks
accuracy exactly as the connector warns (moriio_connector.py:230: "per-layer KV-read barrier
can't fire inside full graph"). => the ~150s stall is COMMON to WRITE and READ, so it is NOT
the WRITE-notify DP-rank mismatch. Reverting READ mode.
NEW HYPOTHESIS: the ~150s is a FIXED COMPUTE/BARRIER cost common to both transfer paths, not
a notify timeout. GPU pinned 100% the whole time. Candidates: (a) decode forward over a
huge/padded batch per request; (b) a MoRI all2all/collective barrier (mori_low_latency
InterNodeV1LL) stalling ~150s; (c) the eager_handshake_all_dp_ranks all-reduce barrier.
Since it's GPU-bound (not network-idle), lean (a)/(b). NEXT: check decode batch/token shape
per request + whether mori_low_latency all2all is the 150s. Consider decode all2all-backend
= mori_high_throughput, or reduce max_num_batched_tokens.

## *** LIKELY REAL FIX: decode CG=PIECEWISE (not FULL_AND_PIECEWISE) ***
Compared to the WORKING v3 (#193) config (~20s/50K): v3 decode = mori_low_latency + CG=**PIECEWISE**
(run_2p2d.sh:100). My v4 runs used decode CG=**FULL_AND_PIECEWISE** (I changed it for "perf").
The FULL graph captures a full max_num_seqs=32 batch; if decode replays the FULL graph for
even a single request, it computes ~32x the work per step -> ~150s. PIECEWISE does not.
Also the connector explicitly warns READ/barrier can't fire inside a FULL graph (:230).
=> The ~150s stall + garbage was likely FULL_AND_PIECEWISE, NOT the transfer path.
FIX: decode CG=PIECEWISE (exact v3 value), WRITE mode, thinking=false. Relaunching. Expect
per-request latency to drop toward v3's ~seconds. This aligns v4 to the proven-working v3
decode config (only the stack underneath is upgraded: base/vLLM/MoRI).
