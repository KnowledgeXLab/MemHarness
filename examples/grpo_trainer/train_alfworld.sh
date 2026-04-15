#!/usr/bin/env bash
set -x
set -euo pipefail
export RAY_ADDRESS='http://10.140.37.75:8265'

ENGINE="vllm" # 须为 vllm（或 hf）；勿用 openai_api 搭配 train_memory_adaptor, vllm | openai_api；openai_api 时需 export OPENAI_API_KEY
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES 2>/dev/null || true
export HYDRA_FULL_ERROR=1
export WANDB_MODE="offline"

# global_pool（Reasoning / Ref / Critic 等）每节点 GPU 数
trainer_n_gpus_per_node=6
# mem_adaptor 专用池每节点 GPU 数（main_ppo 默认与 nnodes 同长度的 1 列表）
mem_adaptor_gpus_per_node=2
# mem_adaptor 环境步调度：与 ppo_trainer.yaml 中 mem_adaptor.env_step_* 一致（减小 early-step 调用、增厚训练样本）
# start=1，end 为开区间（例如 max_steps=30 时用 31 表示允许第 30 步）；every_n=1 表示每步都可触发
mem_adaptor_env_step_start=3
mem_adaptor_env_step_end=31
mem_adaptor_env_step_every_n=3
GPU_NUM="${trainer_n_gpus_per_node}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export MEMADAPTOR_REPO_ROOT="${MEMADAPTOR_REPO_ROOT:-${REPO_ROOT}}"

DATA_ROOT="data/verl-agent"
TRAIN_FILE="${DATA_ROOT}/text/train.parquet"
TEST_FILE="${DATA_ROOT}/text/test.parquet"
VAL_FILE="${TEST_FILE}"

export WANDB_DIR='wandb_logs'

num_cpus_per_env_worker=0.1

TASK_NAME="alfworld"

MEMORY_ENABLED=True
MEMORY_WRITE_BACK=True
EXPERIENCE_SUMMARIZER_MODE="self" # none | self | teacher
RETRIEVAL_MODE="fixed" # agentic | fixed
RETRIEVE_KEY="memory_text" # memory_text | state_text
EMBEDDING_API_URL="http://10.140.37.18:8887/v1"
EMBEDDING_API_KEY="DataFrontier_bge_m3"

MEMORY_REMOTE_SLURM=True
MEMORY_REMOTE_PARTITION="DataFrontier_Explore"
MEMORY_REMOTE_SERVER_PORT="8765"
# 远程起 VDB 的 sbatch：Slurm --exclude，逗号分隔节点名；留空则不排除（见 env.memory.remote_slurm_launch.exclude_nodes）
MEMORY_REMOTE_EXCLUDE_NODES="SH-IDC1-10-140-37-18"
MEMORY_APPTAINER_SIF="/mnt/petrelfs/wurong/glibc_ubuntu22.sif"
MEMORY_CONDA_SH="/mnt/petrelfs/wurong/miniconda3/etc/profile.d/conda.sh"
MEMORY_REMOTE_CONDA_ENV="verl-agent"

# 与 eval 一致：预构建记忆 jsonl；留空则不传参（沿用 yaml 默认 null）
MEMORY_REBUILD_SOURCE_PATH="data/MemAdaptor/AgentTraj-L/${TASK_NAME}_train_memory_records-gpt-5.1.jsonl"
# MEMORY_REBUILD_SOURCE_PATH=""

EXPERIMENT_NAME="train_adaptor"
EXPERIMENTS_ROOT="data/MemAdaptor/exp_results"
MODEL_PATH="models/public_models/Qwen2.5-3B-Instruct"
MEM_ADAPTOR_MODEL_PATH="models/public_models/Qwen2.5-0.5B-Instruct"

if [ "${MEMORY_ENABLED}" = "True" ]; then
  EXPERIMENT_NAME="${EXPERIMENT_NAME}-with_${RETRIEVAL_MODE}_memory"
  EXPERIMENT_NAME="${EXPERIMENT_NAME}-retrieve_${RETRIEVE_KEY}"
  EXPERIMENT_NAME="${EXPERIMENT_NAME}-writeback_${EXPERIENCE_SUMMARIZER_MODE}"
else
  EXPERIMENT_NAME="${EXPERIMENT_NAME}-no_memory"
fi

EXP_DIR="${EXPERIMENTS_ROOT}/${TASK_NAME}/${EXPERIMENT_NAME}"
MEMORY_STORE_DIR="${EXP_DIR}/memory_vdb"

mkdir -p "${EXP_DIR}"
LOG_FILE="${EXP_DIR}/train_alfworld-$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "${LOG_FILE}") 2>&1
echo "[log] Writing full run output to: ${LOG_FILE}"

# 训练 batch：须与 env.rollout.n（GRPO group）及数据量匹配
train_data_size=18
val_data_size=140  ## alfworld验证集只有140条数据，需要整除val_batch_size
group_size=4
max_concurrent=32

# TP 与 GPU 数一致（单卡训练请保持1）
tensor_model_parallel_size=2

VALIDATE_ON_TRAIN_SPLIT=False

PREPARE_FLAGS=()
if [ "${PREPARE_OVERWRITE:-0}" = "1" ] || [ "${PREPARE_OVERWRITE:-}" = "true" ]; then
  PREPARE_FLAGS+=(--overwrite)
fi

python3 -m examples.data_preprocess.prepare \
  --mode 'text' \
  --local_dir "${DATA_ROOT}" \
  --infer_alfworld_sizes \
  --alfworld_eval_split eval_in_distribution \
  "${PREPARE_FLAGS[@]+"${PREPARE_FLAGS[@]}"}"

MEMORY_CLI=()
if [ -n "${MEMORY_REBUILD_SOURCE_PATH}" ]; then
  MEMORY_CLI+=(env.memory.rebuild_source_path="${MEMORY_REBUILD_SOURCE_PATH}")
fi
if [ -n "${EMBEDDING_API_URL}" ]; then
  MEMORY_CLI+=(env.memory.embedding_api_url="${EMBEDDING_API_URL}")
fi
if [ -n "${EMBEDDING_API_KEY}" ]; then
  MEMORY_CLI+=(env.memory.embedding_api_key="${EMBEDDING_API_KEY}")
fi

export VLLM_NCCL_SO_PATH=/mnt/petrelfs/wurong/miniconda3/envs/verl-agent/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2
# Ray Job 里 WorkerDict/vLLM 进程默认拿不到提交机 shell 的 export，须放进 runtime_env.env_vars
RAY_JOB_RUNTIME_ENV_JSON="$(python3 -c "import json, os; print(json.dumps({'excludes': ['logs', 'ray_log', 'swanlog'], 'env_vars': {'VLLM_NCCL_SO_PATH': os.environ['VLLM_NCCL_SO_PATH']}}))")"

REMOTE_VDB_CLI=()
# 接受 True/true/1/yes，避免与 MEMORY_ENABLED 等处写法不一致时静默退回本机 VDB
MEMORY_REMOTE_SLURM_LC="$(printf '%s' "${MEMORY_REMOTE_SLURM:-false}" | tr '[:upper:]' '[:lower:]')"
if [ "${MEMORY_REMOTE_SLURM_LC}" = "true" ] || [ "${MEMORY_REMOTE_SLURM_LC}" = "1" ] || [ "${MEMORY_REMOTE_SLURM_LC}" = "yes" ]; then
  REMOTE_VDB_CLI+=(
    env.memory.remote_slurm_launch.enable=true
    env.memory.remote_slurm_launch.partition="${MEMORY_REMOTE_PARTITION}"
    env.memory.remote_slurm_launch.server_port="${MEMORY_REMOTE_SERVER_PORT}"
    env.memory.server_port="${MEMORY_REMOTE_SERVER_PORT}"
    env.memory.remote_slurm_launch.conda_sh="${MEMORY_CONDA_SH}"
    env.memory.remote_slurm_launch.conda_env="${MEMORY_REMOTE_CONDA_ENV}"
    env.memory.remote_slurm_launch.repo_root="${REPO_ROOT}"
  )
  if [ -n "${MEMORY_APPTAINER_SIF}" ]; then
    REMOTE_VDB_CLI+=(env.memory.remote_slurm_launch.apptainer_sif="${MEMORY_APPTAINER_SIF}")
  fi
  if [ -n "${MEMORY_REMOTE_EXCLUDE_NODES}" ]; then
    REMOTE_VDB_CLI+=(env.memory.remote_slurm_launch.exclude_nodes="${MEMORY_REMOTE_EXCLUDE_NODES}")
  fi
fi

ray job submit --runtime-env-json "${RAY_JOB_RUNTIME_ENV_JSON}" -- \
    python3 -m verl.trainer.main_ppo \
      algorithm.adv_estimator=grpo \
      data.train_files="${TRAIN_FILE}" \
      data.val_files="${VAL_FILE}" \
      data.train_batch_size="${train_data_size}" \
      data.val_batch_size="${val_data_size}" \
      data.max_prompt_length=8192 \
      data.max_response_length=512 \
      data.filter_overlong_prompts=True \
      data.truncation='error' \
      data.return_raw_chat=True \
      actor_rollout_ref.model.path="${MODEL_PATH}" \
      actor_rollout_ref.actor.trainable=false \
      actor_rollout_ref.actor.optim.lr=1e-6 \
      actor_rollout_ref.model.use_remove_padding=True \
      actor_rollout_ref.actor.ppo_mini_batch_size=192 \
      actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
      actor_rollout_ref.actor.use_kl_loss=True \
      actor_rollout_ref.actor.kl_loss_coef=0.01 \
      actor_rollout_ref.actor.kl_loss_type=low_var_kl \
      actor_rollout_ref.model.enable_gradient_checkpointing=True \
      actor_rollout_ref.actor.fsdp_config.param_offload=False \
      actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
      actor_rollout_ref.actor.use_torch_compile=false \
      actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
      actor_rollout_ref.rollout.tensor_model_parallel_size="${tensor_model_parallel_size}" \
      actor_rollout_ref.rollout.name="${ENGINE}" \
      actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
      actor_rollout_ref.rollout.enable_chunked_prefill=False \
      actor_rollout_ref.rollout.enforce_eager=False \
      actor_rollout_ref.rollout.free_cache_engine=False \
      actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
      actor_rollout_ref.rollout.val_kwargs.do_sample=True \
      actor_rollout_ref.rollout.openai_api.max_concurrent="${max_concurrent}" \
      actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
      actor_rollout_ref.ref.fsdp_config.param_offload=True \
      actor_rollout_ref.actor.use_invalid_action_penalty=True \
      actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
      algorithm.use_kl_in_reward=False \
      mem_adaptor.enable=true \
      mem_adaptor.use_actor_rollout_wg=false \
      mem_adaptor.train_memory_adaptor=true \
      mem_adaptor.model.path="${MEM_ADAPTOR_MODEL_PATH}" \
      mem_adaptor.resource_pool_gpus_per_node="[${mem_adaptor_gpus_per_node}]" \
      mem_adaptor.env_step_start="${mem_adaptor_env_step_start}" \
      mem_adaptor.env_step_end="${mem_adaptor_env_step_end}" \
      mem_adaptor.env_step_every_n="${mem_adaptor_env_step_every_n}" \
      env.env_name=alfworld/AlfredTWEnv \
      env.alfworld.validate_on_train_split="${VALIDATE_ON_TRAIN_SPLIT}" \
      env.seed=0 \
      env.max_steps=30 \
      env.memory.enabled="${MEMORY_ENABLED}" \
      env.memory.store_dir="${MEMORY_STORE_DIR}" \
      env.memory.write_back="${MEMORY_WRITE_BACK}" \
      env.memory.experience_summarizer.mode="${EXPERIENCE_SUMMARIZER_MODE}" \
      env.memory.retrieval_mode="${RETRIEVAL_MODE}" \
      env.memory.retrieve_key="${RETRIEVE_KEY}" \
      "${MEMORY_CLI[@]+"${MEMORY_CLI[@]}"}" \
      "${REMOTE_VDB_CLI[@]+"${REMOTE_VDB_CLI[@]}"}" \
      env.rollout.n="${group_size}" \
      env.resources_per_worker.num_cpus="${num_cpus_per_env_worker}" \
      trainer.critic_warmup=0 \
      trainer.logger=['console','wandb'] \
      trainer.project_name='MemAdaptor_alfworld' \
      trainer.experiment_name="${EXPERIMENT_NAME}" \
      trainer.n_gpus_per_node="${trainer_n_gpus_per_node}" \
      trainer.nnodes=1 \
      trainer.save_freq=-1 \
      trainer.test_freq=5 \
      trainer.total_epochs=150 \
      trainer.validation_data_dir="${EXP_DIR}/val_traj" \
      trainer.val_before_train=True \
      trainer.val_only=False \
      "$@"
