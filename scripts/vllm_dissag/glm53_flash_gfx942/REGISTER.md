# Registering GLM-5.3-Flash-MI300 (gfx942) into MAD's disagg harness

MAD uses **dir_models**: each `scripts/<dir>/` has its own `models.json`. The runner
(`madengine` CLI) reads `scripts/vllm_dissag/models.json` + `models.yaml`; nested
per-model-subdir metadata is NOT scanned. So registration = editing these THREE files
(the model's own subdir `glm53_flash_gfx942/` stays docs/scripts/patches only).

The merged `models.json` and `models.yaml` in THIS directory are the reference files
with our model already appended — diff them against upstream to see exactly the 1 added
array entry + 1 added yaml block. Apply those two additions, plus the slurm gate below.

## 1. scripts/vllm_dissag/models.json  (append 1 array entry)
Added entry `pyt_vllm_disagg_mori_glm-5.3-flash_mi300`:
- `dockerfile: ../../docker/vllm_disagg_inference.glmv53flash.mi300` (engine appends `.ubuntu.amd.Dockerfile`)
- `scripts: run_xPyD_models.slurm`, `env_vars.MODEL_NAME: GLM-5.3-Flash-MI300`, `RUN_MORI: 1`
- tags: pyt, vllm, vllm_disagg, mori_ep, inference, gfx942, mi300x, mi325x

## 2. scripts/vllm_dissag/models.yaml  (append 1 map block)
Added `GLM-5.3-Flash-MI300:` — DISTINCT from PR254's `GLM-5.3-Flash` because this
config differs: decode CUDA graph PIECEWISE (not NONE), MNBT 16384, our aiter env. See the
block for the full env + prefill/decode flags.

## 3. scripts/vllm_dissag/run_xPyD_models.slurm  (add to 2 gate arrays)
Add `"GLM-5.3-Flash-MI300"` to BOTH:
- `VALID_MODELS=( ... )`  (~line 81)
- `MORI_EP_VALID_MODELS=( ... )`  (~line 94 — required because our entry sets RUN_MORI=1)
NOTE: the upstream reference `GLM-5.3-Flash` was itself never added to VALID_MODELS (a PR254
omission). Do NOT copy that — our MODEL_NAME must be in both arrays or the slurm aborts.

## Naming consistency (must match across all 3)
MODEL_NAME `GLM-5.3-Flash-MI300` appears identically in: models.json env_vars, the models.yaml
map key, and both slurm gate arrays. The models.json `name`
(`pyt_vllm_disagg_mori_glm-5.3-flash_mi300`) is the engine handle.
