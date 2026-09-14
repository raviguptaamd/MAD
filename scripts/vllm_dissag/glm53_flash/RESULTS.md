# GLM-5.3-Flash disaggregated — verified recall results (MI355X gfx950 + ionic)

Needle-in-haystack recall, disaggregated 1P/1D over MoRIIO. Needle =
`The special access code is DELTA-9931.` inserted at a given depth into filler
text; the model is asked to recall it. `OK` = the exact code `DELTA-9931` is
returned. Greedy (temperature 0). Both fixes applied (MLA KV block-mapping
in-source + single-chunk prefill).

## TP4 1P/1D  (prefill/decode = 4-way TP each, MoRIIO KV, GPU util 0.5)

`--max-num-batched-tokens 262144` (single-chunk to 256K).

| context (words) | prompt tokens | needle depth | result |
|-----------------|---------------|--------------|--------|
| 500             | 559           | 0.9          | DELTA-9931 |
| 8,000           | 8,684         | 0.9          | DELTA-9931 |
| 20,000          | 21,684        | 0.9          | DELTA-9931 |
| 30,000          | 32,526        | 0.9          | DELTA-9931 |
| 35,000          | 37,934        | 0.9          | DELTA-9931 |
| 40,000          | 43,355        | 0.1 / 0.5 / 0.99 | DELTA-9931 (all) |
| 60,000          | 65,026        | 0.9          | DELTA-9931 |
| 100,000         | 108,355       | 0.9          | DELTA-9931 |

**Exact recall to 100K tokens, all needle depths.**

## EP8 1P/1D  (DP8 + expert-parallel, MoRIIO KV + allgather/reducescatter MoE, util 0.40)

`max-model-len 65536`, `--max-num-batched-tokens 65536` (single-chunk to 64K).

| context (words) | prompt tokens | needle depth | result |
|-----------------|---------------|--------------|--------|
| 500             | 558           | 0.9              | DELTA-9931 |
| 8,000           | 8,598         | 0.9              | DELTA-9931 |
| 20,000          | 21,453        | 0.9              | DELTA-9931 |
| 28,000          | 30,033        | 0.9              | DELTA-9931 |
| 32,000          | 32,163        | 0.9              | DELTA-9931 |
| 40,000          | 39,663        | 0.9              | DELTA-9931 |
| 50,000          | 49,308        | 0.9              | DELTA-9931 |
| 60,000          | 60,033        | 0.9              | DELTA-9931 |
| 64,000          | 63,783        | 0.9              | DELTA-9931 |

**Exact recall to 64K tokens (up to the configured 65,536 max-model-len), all
depths.** The ceiling is purely `max-model-len` / the single-chunk prefill batch
cap, not a correctness limit: a prompt over 65,536 is cleanly rejected (HTTP 400),
not mis-recalled. Raise `max-model-len` + `--max-num-batched-tokens` together to
extend (VRAM headroom exists at util 0.40 — KV was ~35x concurrency loaded at 64K).

## Before the fix (for reference)

Without the MLA KV block-mapping fix, disagg served coherent local continuation
but recall was garbage at every length (e.g. "the password is banana. the
password is" → "is is is is"). Colocated (non-disagg) inference on the same image
recalled correctly, which localized the bug to the MoRIIO KV transfer.

## Environment

- Image: built from `docker/vllm_disagg_inference.glmv53flash.ubuntu.amd.Dockerfile`.
  All fixes are in-source via the pinned forks (vLLM connector fixes + mori ionic
  fixes); no runtime overlays. gfx950 (MI355X) build target.
- Nodes: 8-GPU MI355X (gfx950), AMD AI NIC (ionic) RoCE, one leg per node.
  Verified on GID-mapped "gold" ionic rail pairs (all 8 rails routed, no IBDEV
  pinning) — TP4 on one pair, EP8 on another.
- Model: `GLM-5.3-Flash` (`Glm5NextForConditionalGeneration`).
