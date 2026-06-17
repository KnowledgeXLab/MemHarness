#!/usr/bin/env bash
# 启动本地/Slurm 节点上的 vLLM embedding 服务（默认 bge-m3）。
# 建议计算节点用: sbatch scripts/vllm_server.sbatch
# 环境变量（可选）:
#   GPU_NUM      默认 2，与 --tensor-parallel-size 一致
#   VLLM_PORT    默认 8081
#   VLLM_HOST    默认 0.0.0.0（便于其它机器用 http://调度节点给出的IP:端口/v1 访问）
#   VLLM_API_KEY 可选；设置后客户端须在 Authorization: Bearer <key> 中携带（也可 export VLLM_API_KEY）
#   VLLM_NCCL_SO_PATH 未设置则用下面默认路径（verl-agent）
set -euo pipefail

# Slurm sbatch / 外层脚本可传入绝对仓库根（与从脚本路径推断等价，但可避免歧义）
if [[ -n "${MEMADAPTOR_REPO_ROOT:-}" ]]; then
  REPO_ROOT="${MEMADAPTOR_REPO_ROOT}"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "${REPO_ROOT}"

HOSTNAME=$(hostname)
IP=$(hostname -I | awk '{print $1}')

echo "[vllm_server] Repo: ${REPO_ROOT}"
echo "[vllm_server] 服务节点: ${HOSTNAME} (${IP})"
export VLLM_NCCL_SO_PATH="${VLLM_NCCL_SO_PATH:-/mnt/petrelfs/wurong/miniconda3/envs/verl-agent/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2}"

GPU_NUM="${GPU_NUM:-2}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-8081}"
# vLLM 也认环境变量 VLLM_API_KEY；此处允许脚本内 VLLM_API_KEY= 覆盖
VLLM_API_KEY="${VLLM_API_KEY:-DataFrontier_bge_m3}"

VLLM_API_KEY_ARGS=()
if [[ -n "${VLLM_API_KEY}" ]]; then
  VLLM_API_KEY_ARGS=(--api-key "${VLLM_API_KEY}")
  echo "[vllm_server] API key auth: enabled (set Authorization: Bearer <VLLM_API_KEY>)"
else
  echo "[vllm_server] API key auth: disabled (no VLLM_API_KEY)"
fi

# vllm serve models/Qwen/Qwen2.5-0.5B-Instruct --tensor-parallel-size 2 --served-model-name Qwen2.5-0.5b --host 0.0.0.0 --port 8000 --max-model-len 8192 --gpu-memory-utilization 0.7 --uvicorn-log-level debug --disable-log-requests
# vllm serve /mnt/phwfile/datafrontier/wangxiaoman/coldstart/global_step_49 --tensor-parallel-size 2 --served-model-name Qwen2.5-0.5b --host 0.0.0.0 --port 8000 --max-model-len 8192 --gpu-memory-utilization 0.7 --uvicorn-log-level debug --disable-log-requests

exec vllm serve models/public_models/bge-m3 \
  --served-model-name bge_m3 \
  --host "${VLLM_HOST}" \
  --port "${VLLM_PORT}" \
  --tensor-parallel-size "${GPU_NUM}" \
  --max-model-len 8192 \
  --uvicorn-log-level debug \
  --disable-log-requests \
  "${VLLM_API_KEY_ARGS[@]+"${VLLM_API_KEY_ARGS[@]}"}"

# # ## 在vllm1环境启动
# vllm serve models/public_models/Qwen3-VL-8B-Instruct --served-model-name qwen3_vl_8b_instruct --port 8081 --tensor-parallel-size ${GPU_NUM} --max-model-len 8192 --uvicorn-log-level debug --disable-log-requests


# vllm serve /mnt/phwfile/datafrontier/wangxiaoman/models/Qwen/Qwen3-32B --tensor-parallel-size 8 --served-model-name Qwen3-32b --host 0.0.0.0 --port 8000 --max-model-len 32768 --gpu-memory-utilization 0.6 --uvicorn-log-level debug --disable-log-requests


# VLLM_USE_MODELSCOPE=true vllm serve models/public_models/Qwen3-0.6B --reasoning-parser deepseek_r1 --port 8081 --tensor-parallel-size ${GPU_NUM} --max-model-len 8192 --uvicorn-log-level debug --disable-log-requests
