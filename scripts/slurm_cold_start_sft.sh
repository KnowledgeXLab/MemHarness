#!/usr/bin/env bash
#SBATCH --job-name=mem-cold-sft
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-task=4
#SBATCH --partition=DataFrontier_Explore
#SBATCH --output=logs/cold_start/%x-%j.out
#SBATCH --error=logs/cold_start/%x-%j.err

set -euo pipefail

# Slurm 会把本脚本拷到 /var/spool/slurmd/ 再执行，勿用 BASH_SOURCE 推仓库根目录（否则会 mkdir 到 slurmd 目录报 Permission denied）。
# 请在仓库根目录执行：cd /path/to/MemAdaptor && sbatch scripts/slurm_cold_start_sft.sh
# 或事先 export MEMADAPTOR_ROOT=/path/to/MemAdaptor
if [[ -n "${MEMADAPTOR_ROOT:-}" ]]; then
  REPO_ROOT="${MEMADAPTOR_ROOT}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${REPO_ROOT}"
if [[ ! -f "${REPO_ROOT}/scripts/slurm_cold_start_sft.sh" ]]; then
  echo "ERROR: REPO_ROOT=${REPO_ROOT} does not look like MemAdaptor (missing scripts/slurm_cold_start_sft.sh)." >&2
  echo "Fix: cd /path/to/MemAdaptor && sbatch scripts/slurm_cold_start_sft.sh   OR   export MEMADAPTOR_ROOT=/path/to/MemAdaptor" >&2
  exit 1
fi
mkdir -p "${REPO_ROOT}/logs/cold_start"

export HYDRA_FULL_ERROR="1"

export EXP_NAME="webshop-qwen2.5-3b-cold-start-20260511"

# 本地仓库需放在 PYTHONPATH，否则可能找不到 verl / 项目包
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# ---------- 数据与模型 ----------
TRAIN_FILES="data/MemAdaptor/cold_start/webshop/20260511_sample1000_seed42_train.parquet"
VAL_FILES="data/MemAdaptor/cold_start/webshop/20260511_sample100_seed42_val.parquet"
MODEL_ID="models/public_models/Qwen2.5-3B-Instruct"

# 避免多次实验共写一个目录互相覆盖；可按需改成含 SLURM_JOB_ID
SAVE_ROOT="models/save_models/mem_adaptor/cold_start/webshop/${EXP_NAME}"
mkdir -p "${SAVE_ROOT}"

# ---------- 日志----------
export WANDB_MODE="offline"
export WANDB_DIR="./wandb"
mkdir -p "${WANDB_DIR}"
SFT_LOGGER="[wandb,console]"

# 进程数：默认同每节点 GPU 数（若 Slurm 未注入则用手动 NPROC）
if [[ -n "${NPROC:-}" ]]; then
  :
elif [[ -n "${SLURM_GPUS_PER_TASK:-}" ]]; then
  NPROC="${SLURM_GPUS_PER_TASK}"
elif [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
  NPROC="${SLURM_GPUS_ON_NODE}"
else
  NPROC=2
fi


set -x
torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC}" \
  -m verl.trainer.fsdp_sft_trainer \
  data.train_files="${TRAIN_FILES}" \
  data.val_files="${VAL_FILES}" \
  data.multiturn.enable=true \
  data.multiturn.messages_key=messages \
  data.train_batch_size=8 \
  data.micro_batch_size_per_gpu=4 \
  data.max_length=8192 \
  data.truncation=right \
  model.partial_pretrain="${MODEL_ID}" \
  model.enable_gradient_checkpointing=true \
  optim.lr=1e-5 \
  optim.warmup_steps_ratio=0.1 \
  optim.lr_scheduler='cosine' \
  trainer.default_local_dir="${SAVE_ROOT}" \
  trainer.project_name=memadaptor-cold-start \
  trainer.experiment_name="${EXP_NAME}" \
  trainer.logger=['wandb','console'] \
  trainer.total_epochs=2 \
  trainer.default_hdfs_dir=null \
  ulysses_sequence_parallel_size=2 \
  use_remove_padding=true \
  "$@"
