#!/bin/bash
#SBATCH --job-name=e00-bge-m3
#SBATCH --partition=DataFrontier_Explore
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/vllm_server/slurm-%j.out
#SBATCH --error=logs/vllm_server/slurm-%j.err
#SBATCH --quotatype=reserved

# -----------------------------------------------------------------------------
# Embedding vLLM 服务（默认 bge-m3）：通过 sbatch 在计算节点常驻运行。
# 用法（在仓库根目录）:
#   mkdir -p logs/vllm_server
#   sbatch scripts/vllm_server.sbatch
# 按需覆盖分区/时长/GPU 数：
#   sbatch --partition=OtherPart --gres=gpu:4 --time=24:00:00 scripts/vllm_server.sbatch
# 环境变量（可选）:
#   VLLM_CONDA_ENV       默认 verl-agent
#   VLLM_CONDA_SH        默认 /mnt/petrelfs/wurong/miniconda3/etc/profile.d/conda.sh
#   MEMADAPTOR_REPO_ROOT 若设置则直接用（最高优先级）
#   否则批量作业必须用 SLURM_SUBMIT_DIR：提交任务时所在的目录（sbatch 在仓库根执行时即为仓库根）
#   （勿用脚本路径推断仓库根：计算节点上的 batch 拷贝在 /var/spool/slurmd/… 会得到错误路径）
#   GPU_NUM / VLLM_PORT / VLLM_HOST / VLLM_API_KEY —— 传给 scripts/vllm_server.sh（见脚本内注释）
# -----------------------------------------------------------------------------

set -eo pipefail

VLLM_CONDA_SH="${VLLM_CONDA_SH:-/mnt/petrelfs/wurong/miniconda3/etc/profile.d/conda.sh}"
VLLM_CONDA_ENV="${VLLM_CONDA_ENV:-verl-agent}"

if [[ -n "${MEMADAPTOR_REPO_ROOT:-}" ]]; then
  REPO_ROOT="${MEMADAPTOR_REPO_ROOT}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
  REPO_ROOT="$(cd "${THIS_DIR}/.." && pwd)"
fi
cd "${REPO_ROOT}"

export MEMADAPTOR_REPO_ROOT="${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/logs/vllm_server"

echo "[vllm_server.sbatch] SLURM_JOB_ID=${SLURM_JOB_ID:-} NODE=$(hostname)"
echo "[vllm_server.sbatch] REPO_ROOT=${REPO_ROOT} SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>}"
echo "[vllm_server.sbatch] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

if [[ ! -f "${VLLM_CONDA_SH}" ]]; then
  echo "[vllm_server.sbatch] ERROR: conda.sh not found: ${VLLM_CONDA_SH}" >&2
  exit 1
fi
# conda 的 deactivate 脚本会读到未定义的变量；"set -u" 会直接失败
set +u
# shellcheck source=/dev/null
source "${VLLM_CONDA_SH}"
conda activate "${VLLM_CONDA_ENV}"
set -u
set -euo pipefail

exec bash "${REPO_ROOT}/scripts/vllm_server.sh"
