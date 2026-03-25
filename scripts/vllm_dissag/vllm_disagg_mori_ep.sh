#!/bin/bash
# VLLM Disaggregated Server Launcher with Model-Specific Configurations
# =============================================================================

# =============================================================================
# Environment Configuration
# =============================================================================

MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-23731}"
NODE_RANK="${NODE_RANK:-0}"
NNODES="${NNODES:-1}"
MODEL_PATH=$MODEL_PATH
MODEL_NAME="${MODEL_NAME:-}"
xP="${xP:-1}"
yD="${yD:-1}"
IPADDRS="${IPADDRS:-localhost}"
# Comma-separated IPs from Slurm (same order as NODE_RANK). Used by socket_barrier, not for log names.
IFS=',' read -ra IP_ARRAY <<< "${IPADDRS}"

echo "Listing NIXL_COOKBOOK_PATH : "
ls ${NIXL_COOKBOOK_PATH}

# =============================================================================
# Dependencies and Environment Setup
# =============================================================================

pip install py-spy
pip install --ignore-installed --force-reinstall flask


# =============================================================================
# Node-Specific Configuration Maps
# =============================================================================

PREFILL_DP_SIZE=$((xP * 8))
DECODE_DP_SIZE=$((yD * 8))
DP_PARALLEL_SIZE_LOCAL=8
PREFILL_DP_START_RANK=$(( (NODE_RANK - 1) * 8 ))
PREFILL_MASTER_ADDR=$(echo "$IPADDRS" | awk -F',' '{print $2}')
DECODE_DP_START_RANK=$(( (NODE_RANK - xP - 1) * 8 ))
DECODE_MASTER_ADDR=$(echo "$IPADDRS" | awk -F',' -v pos="$xP" '{print $(pos+2)}')
PROXY_PORT=10001

echo "-----------------------------Printing node specific details ----------------------"
echo "IPADDRS = ${IPADDRS}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "HOST_IP=$(hostname -I)"
echo "PREFILL_DP_SIZE=${PREFILL_DP_SIZE}"
echo "DECODE_DP_SIZE=${DECODE_DP_SIZE}"
echo "PREFILL_DP_START_RANK=${PREFILL_DP_START_RANK}"
echo "PREFILL_MASTER_ADDR=${PREFILL_MASTER_ADDR}"
echo "DECODE_DP_START_RANK=${DECODE_DP_START_RANK}"
echo "DECODE_MASTER_ADDR=${DECODE_MASTER_ADDR}"
host_ip=$(hostname -I | awk '{print $1}')
host_name=$(hostname)

echo "Listing NIXL_COOKBOOK_PATH : "
ls ${NIXL_COOKBOOK_PATH}


# =============================================================================
# Container Synchronization
# =============================================================================

echo "Waiting at the container creation barrier on $host_name"
python $NIXL_COOKBOOK_PATH/socket_barrier.py \
    --local-ip ${host_ip} \
    --local-port 2222 \
    --enable-port \
    --node-ips ${IPADDRS} \
    --node-ports 2222

if [ "$NODE_RANK" -eq 0 ]; then
    echo "========= NODE INFO ===================="
    echo "Node list : ${SLURM_JOB_NODELIST}"
    echo "Node IPs  : ${IPADDRS}"
    echo "Model     : ${MODEL_NAME}"
    echo "${host_name}:${host_ip} is Proxy node."

    echo "Proxy server is waiting for prefill & decode nodes to be ready ... "
    sleep 20;

    TIMEOUT_SECONDS=4000
    SLEEP_SECONDS=10
    SEARCH_SIGNAL="Application startup complete."

    # Only NODE_RANK 1 runs the prefill vllm process (see branch below); ranks 2..xP are stubs with no logs.
    # Only NODE_RANK $((xP+1)) runs decode. Waiting for prefill_NODE2..prefill_NODExP or missing decode logs loops forever.
    PREFILL_LOG=/run_logs/${SLURM_JOB_ID}/prefill_NODE1.log
    DECODE_LOG=/run_logs/${SLURM_JOB_ID}/decode_NODE$((xP + 1)).log

    wait_log_signal_or_fail() {
        local LOG_FILE="$1"
        local LABEL="$2"
        local ELAPSED=0
        until grep -q "${SEARCH_SIGNAL}" "${LOG_FILE}" 2>/dev/null; do
            if [ "${ELAPSED}" -ge "${TIMEOUT_SECONDS}" ]; then
                echo "Timeout (${TIMEOUT_SECONDS}s): '${SEARCH_SIGNAL}' not found in ${LABEL}: ${LOG_FILE}" \
                    | tee -a /run_logs/${SLURM_JOB_ID}/proxy_NODE${NODE_RANK}.log
                exit 1
            fi
            sleep "${SLEEP_SECONDS}"
            ELAPSED=$((ELAPSED + SLEEP_SECONDS))
        done
        echo "Ready: ${LABEL} (${LOG_FILE})"
    }

    wait_log_signal_or_fail "${PREFILL_LOG}" "prefill master"
    wait_log_signal_or_fail "${DECODE_LOG}" "decode master"

    sleep 10
    python /app/vllm/examples/online_serving/disaggregated_serving/moriio_toy_proxy_server.py \
        2>&1 | tee -a /run_logs/${SLURM_JOB_ID}/proxy_NODE${NODE_RANK}.log >/dev/null &

    proxy_pid=$!

    # No extra socket_barrier here: wait_log_signal_or_fail already gates on prefill + decode logs.
    # socket_barrier with one port and N IPs waits for that port on *every* node; only rank 1 has
    # 20005 and only rank xP+1 has 40005, so two separate "all nodes on 20005 / 40005" barriers
    # would be wrong or redundant.

    echo "Proxy server ready for benchmarking on ${host_name}:${host_ip}:${PROXY_PORT}"
    sleep 20;
    curl -X POST http://127.0.0.1:10001/v1/completions -H "Content-Type: application/json" -d '{
        "prompt": "Who is AMD CEO?",
        "temperature": 0,
        "max_tokens" : 10,
        "top_k": 1
    }'

    echo "Killing the proxy server.."
    kill $proxy_pid;

elif [ "$NODE_RANK" -eq 1 ]; then
    echo "========= NODE INFO ===================="
    echo "Node list : ${SLURM_JOB_NODELIST}"
    echo "Node IPs  : ${IPADDRS}"
    echo "Model     : ${MODEL_NAME}"
    echo "${host_name}:${host_ip} is Prefill master node."
    echo "PREFILL_DP_SIZE=${PREFILL_DP_SIZE}"
    echo "PREFILL_START_RANK=${PREFILL_START_RANK}"
    echo "PREFILL_MASTER_ADDR=${PREFILL_MASTER_ADDR}"
    echo "DP_PARALLEL_SIZE_LOCAL=${DP_PARALLEL_SIZE_LOCAL}"

    export VLLM_ROCM_USE_AITER=1
    export VLLM_ROCM_USE_AITER_MOE=1
    export VLLM_LOGGING_LEVEL=INFO
    export VLLM_USE_V1=1
    export VLLM_ROCM_USE_AITER_MLA=1
    export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=0
    export VLLM_ALL2ALL_BACKEND=mori
    export GLOO_SOCKET_IFNAME=eth0
    export VLLM_ENGINE_READY_TIMEOUT_S=3600

    vllm serve ${MODEL_PATH} \
        -tp 1 \
        --data-parallel-size ${PREFILL_DP_SIZE} \
        --data-parallel-size-local ${DP_PARALLEL_SIZE_LOCAL} \
        --data-parallel-address ${PREFILL_MASTER_ADDR} \
        --data-parallel-rpc-port 13345 \
        --api-server-count=8 \
        --enable-expert-parallel \
        --port 20005 \
        --gpu_memory_utilization 0.8 \
        --kv-cache-dtype fp8 \
        --block-size 1 \
        --no-enable-prefix-caching \
        --all2all-backend mori \
        --trust-remote-code \
        --enforce-eager \
        --kv-transfer-config '{"kv_connector":"MoRIIOConnector","kv_role":"kv_producer","kv_port":"9711","kv_connector_extra_config":{"proxy_ip":"'"${MASTER_ADDR}"'","proxy_port":"10001","proxy_ping_port":"36367","http_port":"20005","local_ping_port":"61555","handshake_port":"8405","notify_port":"61005"}}' \
        2>&1 | tee /run_logs/${SLURM_JOB_ID}/prefill_NODE${NODE_RANK}.log > /dev/null & 

    prefill_master_pid=$!
    
    echo "Waiting for proxy server to be up..."
    python $NIXL_COOKBOOK_PATH/socket_barrier.py \
        --node-ips ${MASTER_ADDR} \
        --node-ports $PROXY_PORT

    echo "Waiting untill proxy server closes..."
    python $NIXL_COOKBOOK_PATH/socket_wait.py \
        --remote-ip ${MASTER_ADDR} \
        --remote-port $PROXY_PORT

    echo "Killing the prefill master server"
    kill $prefill_master_pid

elif [ "$NODE_RANK" -gt 1 ] && [ "$NODE_RANK" -le "$xP" ]; then
    echo "Prefill child nodes."
    echo "========= NODE INFO ===================="
    echo "Node list : ${SLURM_JOB_NODELIST}"
    echo "Node IPs  : ${IPADDRS}"
    echo "Model     : ${MODEL_NAME}"
    echo "${host_name}:${host_ip} is Prefill master node."
    echo "PREFILL_DP_SIZE=${PREFILL_DP_SIZE}"
    echo "PREFILL_START_RANK=${PREFILL_START_RANK}"
    echo "PREFILL_MASTER_ADDR=${PREFILL_MASTER_ADDR}"
    echo "DP_PARALLEL_SIZE_LOCAL=${DP_PARALLEL_SIZE_LOCAL}"

    export VLLM_ROCM_USE_AITER=1
    export VLLM_ROCM_USE_AITER_MOE=1
    export VLLM_LOGGING_LEVEL=INFO
    export VLLM_USE_V1=1
    export VLLM_ROCM_USE_AITER_MLA=1
    export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=0
    export VLLM_ALL2ALL_BACKEND=mori
    export GLOO_SOCKET_IFNAME=eth0
    export VLLM_ENGINE_READY_TIMEOUT_S=3600

    

elif [ "$NODE_RANK" -eq $((xP + 1))  ]; then
    echo "========= NODE INFO ===================="
    echo "Node list : ${SLURM_JOB_NODELIST}"
    echo "Node IPs  : ${IPADDRS}"
    echo "Model     : ${MODEL_NAME}"
    echo "${host_name}:${host_ip} is Decode master node."
    echo "DECODE_DP_SIZE=${DECODE_DP_SIZE}"
    echo "DECODE_START_RANK=${DECODE_START_RANK}"
    echo "DECODE_MASTER_ADDR=${DECODE_MASTER_ADDR}"
    echo "DP_PARALLEL_SIZE_LOCAL=${DP_PARALLEL_SIZE_LOCAL}"


    export VLLM_ROCM_USE_AITER=1
    export VLLM_ROCM_USE_AITER_MOE=1
    export VLLM_LOGGING_LEVEL=INFO
    export VLLM_USE_V1=1
    export VLLM_ROCM_USE_AITER_MLA=1
    export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=0
    export VLLM_ALL2ALL_BACKEND=mori
    export GLOO_SOCKET_IFNAME=eth0
    export VLLM_ENGINE_READY_TIMEOUT_S=3600

    vllm serve ${MODEL_PATH} \
        -tp 1 \
        --data-parallel-size ${DECODE_DP_SIZE} \
        --data-parallel-size-local ${DP_PARALLEL_SIZE_LOCAL} \
        --data-parallel-address ${DECODE_MASTER_ADDR} \
        --data-parallel-rpc-port 13345 \
        --api-server-count=8 \
        --enable-expert-parallel \
        --port 20005 \
        --gpu_memory_utilization 0.8 \
        --kv-cache-dtype fp8 \
        --block-size 1 \
        --no-enable-prefix-caching \
        --all2all-backend mori \
        --trust-remote-code \
        --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "custom_ops": ["+quant_fp8"]}' \
        --kv-transfer-config '{"kv_connector":"MoRIIOConnector","kv_role":"kv_consumer","kv_port":"9711","kv_connector_extra_config":{"proxy_ip":"'"${MASTER_ADDR}"'","proxy_port":"10001","proxy_ping_port":"36367","http_port":"20005","local_ping_port":"61555","handshake_port":"8405","notify_port":"61005"}}' \
        2>&1 | tee /run_logs/${SLURM_JOB_ID}/decode_NODE${NODE_RANK}.log > /dev/null & 

    decode_master_pid=$!
    
    echo "Waiting for proxy server to be up..."
    python $NIXL_COOKBOOK_PATH/socket_barrier.py \
        --node-ips ${MASTER_ADDR} \
        --node-ports $PROXY_PORT

    echo "Waiting untill proxy server closes..."
    python $NIXL_COOKBOOK_PATH/socket_wait.py \
        --remote-ip ${MASTER_ADDR} \
        --remote-port $PROXY_PORT

    echo "Killing the decode master server"
    kill $decode_master_pid

else
    echo "Decode child nodes..."

fi

echo "Script completed successfully."
exit 0
