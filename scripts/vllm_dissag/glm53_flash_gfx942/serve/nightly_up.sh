#!/bin/bash
# Bring up ONE long-lived vllm-nightly container (sleep infinity), then serve GLM-5.3-Flash-FP8 TP4 eager.
# Run ON the compute node. Weights stay hot (container persists); iterate via docker exec.
set -u
NAME=${NAME:-nite}
IMG=vllm/vllm-openai-rocm:nightly
CACHE=/mnt/m2m_nobackup/ravgupta/glm53_cache/nightly
mkdir -p "$CACHE"
docker rm -f "$NAME" >/dev/null 2>&1
docker run -d --name "$NAME" --network host --ipc host --privileged --group-add video \
  --device /dev/kfd --device /dev/dri --cap-add IPC_LOCK --shm-size 128G --ulimit memlock=-1:-1 \
  -v /mnt/m2m_nobackup/ravgupta/glm53_models:/models:ro -v "$CACHE:/opt/vllm_cache" \
  -e HIP_VISIBLE_DEVICES=0,1,2,3 -e VLLM_USE_V1=1 \
  --entrypoint bash "$IMG" -lc "sleep infinity"
echo "container $NAME up on $(hostname -s)"
