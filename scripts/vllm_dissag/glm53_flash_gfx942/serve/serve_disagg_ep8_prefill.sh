#!/bin/bash
# GLM-5.3-Flash-FP8 EP8 DISAGG PREFILL leg (kv_producer). Runs INSIDE nite on 030.
# Prefill: EAGER. proxy_ip = this node 030 = 10.158.215.107 ; proxy_ping_port=36367.
set -u
M=/models/GLM-5.3-Flash-FP8
LOG=/opt/vllm_cache/disagg_prefill.log

# --- JIT cache persistence (do NOT override AITER_JIT_DIR: the image ships a
#     prebuilt module_aiter_core.so; pointing it at an empty dir forces a rebuild
#     that FAILS. /opt/vllm_cache is already a host bind-mount, so triton/vllm
#     caches persist there across restarts with the image defaults.) ---
export TRITON_CACHE_DIR=/opt/vllm_cache/triton_cache
export VLLM_CACHE_ROOT=/opt/vllm_cache/vllm
mkdir -p "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT"

export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_USE_V1=1
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MLA=1
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_ROCM_USE_AITER_RMSNORM=1
export VLLM_ROCM_USE_AITER_FP8BMM=0
export VLLM_USE_DEEP_GEMM=0
export VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=4096

# --- mlx5 MoRIIO fabric ---
export RDMA_DEVICES=mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9
export MORI_IB_HCA="$RDMA_DEVICES"
export MORI_RDMA_DEVICES="$RDMA_DEVICES"
export MORI_IB_GID_INDEX=3
export MORI_SOCKET_IFNAME=eth0
export NCCL_IB_HCA="$RDMA_DEVICES"
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export MORI_IO_DISABLE_ATOMIC_MR=1
export MORI_NO_ATOMIC_MR=1
export MORI_ENABLE_DMABUF_REG=1
export MORI_SHMEM_HEAP_SIZE=8589934592
export RDMAV_FORK_SAFE=1
export VLLM_HANDSHAKE_TIMEOUT_MINS=30
export VLLM_MORIIO_TRANSFER_TIMEOUT_S=600

# --- long-context unlock (config-only, PROVEN EP8/EP8 disagg over MoRIIO): cap
#     per-forward-pass token count below the MML-scaled int32 workspace-offset
#     ceiling, and disable prefix caching (its partial re-prefill path faults at
#     56-64K). Two coupled knobs: (a) higher MML shrinks the per-pass ceiling;
#     (b) lower MNBT keeps each pass under it. MEASURED EP8/EP8 NIAH (needle
#     74923, 3x): MML=270000 MNBT=16384 -> 64K + 100K + 256K(256347 tok) all
#     PASS 3/3 over the disagg transfer. (MML=110000 MNBT=24576 also does 100K.)
#     Env-overridable to re-sweep. ---
MML="${MML:-270000}"
MNBT="${MNBT:-16384}"

KV='{"kv_connector":"MoRIIOConnector","kv_role":"kv_producer","kv_port":"9711","kv_connector_extra_config":{"proxy_ip":"10.158.215.107","proxy_port":"10001","proxy_ping_port":"36367","http_port":"20005","local_ping_port":"61555","handshake_port":"8405","notify_port":"61005","read_mode":"true"}}'

echo "=== EP8 DISAGG PREFILL (EAGER) $(date) ===" > $LOG
# Kill any prior vllm and WAIT for VRAM to actually release (zombies hold HBM;
# pkill alone races the next launch -> "Free memory less than desired"). Poll
# until GPU0 used < 5 GiB or 60s elapse.
pkill -9 -f "vllm serve" 2>/dev/null; sleep 3
for i in $(seq 1 30); do
  U=$(rocm-smi --showmeminfo vram 2>/dev/null | grep -i used | head -1 | awk '{print $NF}')
  [ -z "$U" ] && break
  GB=$(( U / 1000000000 ))
  [ "$GB" -lt 5 ] && break
  sleep 2
done
echo "VRAM_GB_before_launch=${GB:-unknown}" | tee -a $LOG

nohup vllm serve "$M" \
  --data-parallel-size 8 \
  --enable-expert-parallel \
  --all2all-backend allgather_reducescatter \
  --trust-remote-code \
  --port 20005 \
  --gpu-memory-utilization 0.6 \
  --max-model-len "$MML" \
  --max-num-batched-tokens "$MNBT" \
  --no-enable-prefix-caching \
  --block-size 4 \
  --kv-cache-dtype auto \
  --enforce-eager \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --enable-auto-tool-choice \
  --skip-mm-profiling \
  --kv-transfer-config "$KV" \
  >> $LOG 2>&1 &
echo "prefill_pid=$!"
sleep 3
echo "--- log head ---"; head -30 $LOG
