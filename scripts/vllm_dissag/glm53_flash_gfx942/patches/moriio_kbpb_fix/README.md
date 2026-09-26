# GLM-5.3-Flash disagg KV-transfer recall fix — INDEXER_KBPB

Fixes the ~4096-token recall wall in EP8/EP8 1P/1D disaggregated serving over
MoRIIO (mlx5 RoCEv2, gfx942 MI300X). Before this fix, recall over the P->D KV
transfer corrupted beyond one attention block (spec block_size 4352 tok ~ 4096);
the identical prompt sent DIRECT to the prefill leg (no transfer) recalled fine.

## Root cause (pinpointed with MORIIO_OFFSET_DBG=1 instrumentation)

GLM-5.3-Flash has HYBRID KV: MLA `self_attn.attn` + a DSA sparse-indexer
`self_attn.indexer.k_cache` + `tail_cache`. vLLM's hybrid KV allocator hands out
block-ids in a single shared GROUP-block unit (decode ref = 1012 group blocks,
prefill ref = 1139). Most caches are paged 1:1 at that unit, so a group block-id
N maps directly to kernel block N. BUT the indexer.k_cache is paged at a finer
kernel block:

    indexer.k_cache decode shape = (17204, 1, 64, 132)  = 1012 * 17 kernel blocks
    indexer.k_cache prefill shape = (19363, ...)         = 1139 * 17 kernel blocks

So ONE group block-id N actually spans kbpb = 17 contiguous kernel sub-blocks
[N*17 .. N*17+16]. The native connector transferred the RAW group id as a kernel
id, so every group block beyond #0 (i.e. every token past 4352 ~ the 4096 wall)
read the indexer's KV from the WRONG kernel offset (block 1 instead of 17..33) ->
sparse-indexer rankings garbage -> recall corruption. All OTHER layers have
kbpb == 1 and were already correct, which is why the wall sat exactly at one
attention block.

## The fix

`moriio_layout.py :: compute_block_transfer_offsets` — for single-region
(transfers_per_block == 1) MLA/hybrid layers, derive
  kbpb = per_layer_num_blocks / reference_group_block_count
(per leg: local uses this rank's num_blocks=1012; remote uses the peer's
num_blocks=1139, both from the MoRIIO handshake metadata) and expand each group
block-id into its kbpb kernel sub-blocks BEFORE computing byte offsets. Local and
remote share the SAME group-id list and SAME ratio (17), so per-side expansion
stays aligned. kbpb == 1 (every non-indexer layer) is a byte-identical no-op.

`moriio_connector.py :: _compute_block_transfer_offsets` — passes the two
reference group-block counts: `local_num_blocks=self.num_blocks`,
`remote_ref_blocks=remote_moriio_meta.num_blocks`.

This is the same class of fix as MAD PR254's `_mla_kernel_blocks_per_group_block`,
but adapted to our NEWER connector (4D packed MLA caches with num_heads/num_states
AttentionSpec) instead of PR254's older 3D-cache API. A wholesale PR254 overlay
REGRESSED our stack (garbage at 24 tokens) because of that API skew, so only the
block-routing idea was ported.

## Verified (needle 74923, depth 0.5, through the MoRIIO proxy :10001)

Before: 3379 tok PASS, 4489 tok FAIL (fresh-chat/garbage).
After:  4489, 6709, 8929, 17819, 35599, 71159 prompt tok ALL PASS.
KBPB debug confirmed firing ONLY on indexer.k_cache (kbpb=17), all else kbpb=1.

100K crashed a decode EP worker (gloo peer-closed / EngineDeadError) — a SEPARATE
scaling fault, NOT the transfer recall bug (which is fixed through 64K).

## Deploy

Bind-copy `moriio_layout.py` + `moriio_connector.py` over the in-container native
files (engine/common are unchanged native, included for a complete set), clear
__pycache__, `docker restart nite`, relaunch. Hardware config (mlx5, GID=3,
atomic-MR strip) stays in the serve scripts' env — these Python files are
hardware-independent (cache geometry + block accounting only).
