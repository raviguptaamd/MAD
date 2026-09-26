#!/bin/bash
# GLM-5.3-Flash-FP8 TP4/TP4 DISAGG PREFILL leg (kv_producer). Runs INSIDE nite on node 015.
# Prefill: EAGER, TP4, GPUs 0-3. proxy runs on THIS node 015 = 10.158.213.1 ; proxy_ping_port=37367.
# Derived from serve/serve_disagg_ep8_prefill.sh: EP8 -> TP4 (removed DP8/expert-parallel/all2all,
# added -tp 4), HIP 0-3, ports shifted +1000 from EP8 to avoid the live EP8 pair (030/038).
set -u
M=/models/GLM-5.3-Flash-FP8
LOG=/opt/vllm_cache/tp4_prefill.log

# --- JIT cache persistence (do NOT override AITER_JIT_DIR) ---
export TRITON_CACHE_DIR=/opt/vllm_cache/triton_cache
export VLLM_CACHE_ROOT=/opt/vllm_cache/vllm
mkdir -p "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT"

# --- DP knob: DP=1 -> TP4/TP4 (4 GPU, GPUs 0-3); DP=2 -> TP4xDP2 (8 GPU, GPUs 0-7,
#     recall to 256K, escapes the 64K fault). Router RDP must match (1 or 2), and for
#     DP=2 launch the router with --moriio-dp-size 2. ---
DP="${DP:-1}"
if [ "$DP" = "2" ]; then
  export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  DP_FLAG="--data-parallel-size 2"
else
  export HIP_VISIBLE_DEVICES=0,1,2,3
  DP_FLAG=""
fi
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

# --- long-context unlock (proven EP8/EP8): MML high, MNBT low ---
MML="${MML:-270000}"
MNBT="${MNBT:-16384}"

# KBPB debug on so the decode leg's [KBPB]/[OFFDBG] lines are captured for the
# TP4 factor read-out (prefill logs the producer side; decode logs the consumer).
export MORIIO_OFFSET_DBG="${MORIIO_OFFSET_DBG:-1}"

# proxy_ip = THIS node 015 = 10.158.213.1 ; ports +1000 from EP8 (kv_port 9711->10711,
# proxy_port 10001->11001, proxy_ping 36367->37367, http 20005->21005,
# local_ping 61555->62555, handshake 8405->9405, notify 61005->62005).
KV='{"kv_connector":"MoRIIOConnector","kv_role":"kv_producer","kv_port":"10711","kv_connector_extra_config":{"proxy_ip":"10.158.213.1","proxy_port":"11001","proxy_ping_port":"37367","http_port":"21005","local_ping_port":"62555","handshake_port":"9405","notify_port":"62005","read_mode":"true"}}'

echo "=== TP4 DISAGG PREFILL (EAGER, GPU0-3) $(date) ===" > $LOG
# Scope the kill to THIS leg's port only (2-node, but keep the discipline).
pkill -9 -f "vllm serve.*--port 21005" 2>/dev/null; sleep 3
for i in $(seq 1 30); do
  U=$(rocm-smi --showmeminfo vram 2>/dev/null | grep -i used | head -1 | awk '{print $NF}')
  [ -z "$U" ] && break
  GB=$(( U / 1000000000 ))
  [ "$GB" -lt 5 ] && break
  sleep 2
done
echo "VRAM_GB_before_launch=${GB:-unknown}" | tee -a $LOG

nohup vllm serve "$M" \
  --tensor-parallel-size 4 \
  $DP_FLAG \
  --trust-remote-code \
  --port 21005 \
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
