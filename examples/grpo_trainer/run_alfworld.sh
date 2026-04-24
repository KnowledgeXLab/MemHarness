#!/usr/bin/env bash
# AlfWorld GRPO **基线**：无外部记忆、无 format_reward，仅环境 episode 回报 + GRPO。
# 用于与 ``train_alfworld-evolver.sh``（记忆 + format）对照；需集群 vLLM rollout。
# 用法：./run_alfworld.sh [vllm|hf]   默认 vllm
# 与 evolver 相同：``ray job submit`` 提交到 ``RAY_ADDRESS``（Ray Dashboard HTTP 地址，端口多为 8265）。
set -x
set -euo pipefail

# 与 train_alfworld-evolver.sh 一致；提交前可 ``export RAY_ADDRESS=...`` 覆盖。
export RAY_ADDRESS="http://10.140.37.99:8265"

ENGINE="vllm"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES 2>/dev/null || true
export HYDRA_FULL_ERROR=1
export WANDB_MODE="${WANDB_MODE:-offline}"

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
MEMORY_ENABLED=False

# --- 与 ppo_trainer.yaml 对齐：记忆相关默认能关的全显式关掉（避免仅依赖 enabled=False 的边界行为）---
MEMORY_BASELINE_CLI=(
  env.memory.enabled="${MEMORY_ENABLED}"
  env.memory.auto_start_server=false
  env.memory.write_back=false
  env.memory.remote_slurm_launch.enable=false
  env.memory.retrieval_mode_phases=null
  env.memory.rebuild_source_path=null
  env.memory.debug_retrieval=false
  env.memory.experience_utility.enable=false
  env.memory.experience_utility.prune_every_n_global_steps=0
)

# --- 基线：关闭 format_reward（yaml 默认 enable=true，必须显式关）---
FORMAT_REWARD_ENABLE=False

EXPERIMENT_NAME="grpo_baseline_no_memory_no_format"
EXPERIMENTS_ROOT="data/MemAdaptor/exp_results"
# 与仓库内路径一致；可 export MODEL_PATH=... 覆盖
MODEL_PATH="models/public_models/Qwen2.5-1.5B-Instruct"

EXP_DIR="${EXPERIMENTS_ROOT}/${TASK_NAME}/${EXPERIMENT_NAME}"
PLACEHOLDER_MEMORY_STORE="${EXP_DIR}/memory_vdb_unused"

mkdir -p "${EXP_DIR}" "${PLACEHOLDER_MEMORY_STORE}"
LOG_FILE="${EXP_DIR}/run_alfworld_baseline-$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "${LOG_FILE}") 2>&1
echo "[log] Writing full run output to: ${LOG_FILE}"

train_data_size=16
val_data_size=140
group_size=8

tensor_model_parallel_size=2
trainer_n_gpus_per_node=8

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

FORMAT_REWARD_CLI=(
  reward_model.format_reward.enable="${FORMAT_REWARD_ENABLE}"
)

# Ray Job 里 WorkerDict/vLLM 进程默认拿不到提交机 shell 的 export，须放进 runtime_env.env_vars
VLLM_NCCL_SO_PATH="${VLLM_NCCL_SO_PATH:-/mnt/petrelfs/wurong/miniconda3/envs/verl-agent/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2}"
export VLLM_NCCL_SO_PATH
RAY_JOB_RUNTIME_ENV_JSON="$(python3 -c "import json, os; print(json.dumps({'excludes': ['logs', 'ray_log', 'swanlog'], 'env_vars': {'VLLM_NCCL_SO_PATH': os.environ['VLLM_NCCL_SO_PATH']}}))")"

ray job submit --runtime-env-json "${RAY_JOB_RUNTIME_ENV_JSON}" -- \
    python3 -m verl.trainer.main_ppo \
      algorithm.adv_estimator=grpo \
      data.train_files="${TRAIN_FILE}" \
      data.val_files="${VAL_FILE}" \
      data.train_batch_size="${train_data_size}" \
      data.val_batch_size="${val_data_size}" \
      data.max_prompt_length=2048 \
      data.max_response_length=512 \
      data.filter_overlong_prompts=True \
      data.truncation='error' \
      data.return_raw_chat=True \
      actor_rollout_ref.model.path="${MODEL_PATH}" \
      actor_rollout_ref.actor.optim.lr=1e-6 \
      actor_rollout_ref.model.use_remove_padding=True \
      actor_rollout_ref.actor.ppo_mini_batch_size=256 \
      actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
      actor_rollout_ref.actor.use_kl_loss=True \
      actor_rollout_ref.actor.kl_loss_coef=0.01 \
      actor_rollout_ref.actor.kl_loss_type=low_var_kl \
      actor_rollout_ref.model.enable_gradient_checkpointing=True \
      actor_rollout_ref.actor.fsdp_config.param_offload=False \
      actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
      actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
      actor_rollout_ref.rollout.tensor_model_parallel_size="${tensor_model_parallel_size}" \
      actor_rollout_ref.rollout.name="${ENGINE}" \
      actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
      actor_rollout_ref.rollout.enable_chunked_prefill=False \
      actor_rollout_ref.rollout.enforce_eager=False \
      actor_rollout_ref.rollout.free_cache_engine=False \
      actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
      actor_rollout_ref.rollout.val_kwargs.do_sample=True \
      actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
      actor_rollout_ref.ref.fsdp_config.param_offload=True \
      actor_rollout_ref.actor.use_invalid_action_penalty=True \
      actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
      algorithm.use_kl_in_reward=False \
      mem_adaptor.enable=false \
      mem_adaptor.train_memory_adaptor=false \
      env.env_name=alfworld/AlfredTWEnv \
      env.alfworld.validate_on_train_split="${VALIDATE_ON_TRAIN_SPLIT}" \
      env.seed=0 \
      env.max_steps=50 \
      env.memory.store_dir="${PLACEHOLDER_MEMORY_STORE}" \
      "${MEMORY_BASELINE_CLI[@]+"${MEMORY_BASELINE_CLI[@]}"}" \
      "${FORMAT_REWARD_CLI[@]+"${FORMAT_REWARD_CLI[@]}"}" \
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
      trainer.val_before_train=False \
      trainer.val_only=False \
      "$@"
