#!/bin/bash
# GLM-5.3-Flash-FP8 TP4/TP4 DISAGG DECODE leg (kv_consumer). Runs INSIDE nite on node 028.
# Decode: PIECEWISE cudagraph + MTP(1) + --max-num-seqs 64, TP4, GPUs 0-3.
# proxy_ip = PREFILL/proxy node 015 = 10.158.213.1 ; proxy_ping_port=37367.
# Derived from serve/serve_disagg_ep8_decode.sh: EP8 -> TP4 (removed DP8/expert-parallel/all2all,
# added -tp 4), HIP 0-3, ports shifted +1000 from EP8. MTP added per task (EP8 serve/ script had
# it stripped in a baseline variant; STACK.md mandates MTP on the decode leg).
set -u
M=/models/GLM-5.3-Flash-FP8
LOG=/opt/vllm_cache/tp4_decode.log

# --- JIT cache persistence (do NOT override AITER_JIT_DIR) ---
export TRITON_CACHE_DIR=/opt/vllm_cache/triton_cache
export VLLM_CACHE_ROOT=/opt/vllm_cache/vllm
mkdir -p "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT"

# --- TP4 uses 4 GPUs (this is a dedicated decode node; GPUs 0-3) ---
export HIP_VISIBLE_DEVICES=0,1,2,3
export VLLM_USE_V1=1
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MLA=1
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_ROCM_USE_AITER_RMSNORM=1
export VLLM_ROCM_USE_AITER_FP8BMM=0
export VLLM_USE_DEEP_GEMM=0
export VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=4096
export VLLM_USE_BREAKABLE_CUDAGRAPH=1   # decode cudagraph correctness (indexer outside graph)

# --- mlx5 MoRIIO fabric (RoCEv2 GID3) ---
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

# KBPB debug on so the [KBPB]/[OFFDBG] consumer-side lines are captured for the
# TP4 factor read-out (this is the key diagnostic the task asks for).
export MORIIO_OFFSET_DBG="${MORIIO_OFFSET_DBG:-1}"

# proxy_ip = PREFILL/proxy node 015 = 10.158.213.1 ; ports +1000 from EP8
# (kv_port 6301->7301, proxy_port 10001->11001, proxy_ping 36367->37367,
# http 40005->41005, local_ping 4583->5583, handshake 7305->8305, notify 61005->62005).
# 2-node pair: decode is on a different host than prefill, so notify_port matches
# prefill's (62005) exactly as the EP8 pair does (no single-node collision here).
KV='{"kv_connector":"MoRIIOConnector","kv_role":"kv_consumer","kv_port":"7301","kv_connector_extra_config":{"proxy_ip":"10.158.213.1","proxy_port":"11001","proxy_ping_port":"37367","http_port":"41005","local_ping_port":"5583","handshake_port":"8305","notify_port":"62005","read_mode":"true"}}'

echo "=== TP4 DISAGG DECODE (PIECEWISE + MTP, GPU0-3) $(date) ===" > $LOG
pkill -9 -f "vllm serve.*--port 41005" 2>/dev/null; sleep 3
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
  --trust-remote-code \
  --port 41005 \
  --gpu-memory-utilization 0.6 \
  --max-model-len "$MML" \
  --max-num-batched-tokens "$MNBT" \
  --max-num-seqs 64 \
  --no-enable-prefix-caching \
  --block-size 4 \
  --kv-cache-dtype auto \
  --compilation-config '{"cudagraph_mode":"PIECEWISE"}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --enable-auto-tool-choice \
  --skip-mm-profiling \
  --kv-transfer-config "$KV" \
  >> $LOG 2>&1 &
echo "decode_pid=$!"
sleep 3
echo "--- log head ---"; head -30 $LOG
