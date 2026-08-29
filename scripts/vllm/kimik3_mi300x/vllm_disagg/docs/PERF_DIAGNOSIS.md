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

## PIECEWISE test RESULT: also 171.9s (correct recall). Stall is INVARIANT.
decode CG=PIECEWISE (exact v3 value): request 171.9s, recall CORRECT. So the ~150-170s is
INVARIANT across: WRITE/READ mode, F_A_P/PIECEWISE cudagraph. Not the transfer path, not
cudagraph, not reasoning, not DP-rank mismatch.

## PRECISE TRACE of the stall (both sides 100% GPU, no logs, ~150s):
For a matched request (prefill Worker_DP1 + decode notify rank=1 -- ranks AGREE):
  01:32:08 prefill DP1 "write-stash" (KV staged, moriio_connector.py:2730)
  01:32:10 decode "notify prefill rank=1" (blocks ready) -- handshake+notify FAST (~2s), MATCHED
  01:32:10 -> ~01:34:40 : ZERO log activity on decode AND prefill; BOTH GPUs pinned 100%;
            ~150s later the request completes with correct output.
=> After a correct+fast handshake, BOTH prefill(rank1) and decode spin at 100% GPU for ~150s
   with no logs. This is a COMPUTE/COLLECTIVE stall, not a network timeout (network idle would
   be GPU~0). Signature = a MoRI all2all collective barrier on the decode MoE path
   (mori_low_latency = InterNodeV1LL cross-node dispatch/combine) hanging ~150s per request,
   OR a decode forward recomputing something huge. Per-request (req2 also ~178s), not one-time.

## RULED OUT this session: reasoning(thinking off=same), WRITE-notify DP mismatch(READ=same
## 150s+garbage), cudagraph mode(PIECEWISE=same 171s), qp/num_workers knobs(broke transfer).
## REMAINING SUSPECTS: (1) MoRI InterNodeV1LL decode all2all per-request ~150s barrier on
## bnxt (new MoRI 624002c8 vs v3's older MoRI -- THE version delta); (2) decode all2all-backend
## mori_high_throughput instead of low_latency; (3) MoRI bisect. v3 used older MoRI + same
## mori_low_latency and got ~20s/50K -> the MoRI version is the prime suspect.

## *** MAJOR REFRAME: the ~150s AMORTIZES across concurrency (it batches!) ***
Live serve (decode PIECEWISE+LL), thinking=false, 2K reqs:
- N=1: ~170s
- N=4 concurrent: wall=181s, ALL 4 complete, ALL recall correct.
=> 4 requests in ~the time of 1. The ~150s is NOT per-request-serial -- it's a FIXED
   per-decode-wave latency FLOOR that is SHARED across all in-flight requests. Effective
   per-request cost at con=4 = ~45s; expected to keep dropping with concurrency.
This means the serve IS throughput-capable; the ~150s is a fixed floor (a decode-step /
all2all warmup or a fixed wait that fires once per wave), amortized by batching. Far more
optimizable than a serial bug. The floor itself is still worth killing (latency), but
THROUGHPUT is fine and scales with con. Confirm con=16 amortization next; then chase the
floor (MoRI all2all warmup / a fixed per-step barrier).
