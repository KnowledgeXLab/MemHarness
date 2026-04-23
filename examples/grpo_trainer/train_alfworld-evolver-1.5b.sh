#!/usr/bin/env bash
# EvolveR 式「在线交互」阶段（本仓库实现）：在 **带记忆检索/写回** 的 AlfWorld 上，
# 用 GRPO **只训练主 Reasoning policy**（actor_rollout_ref），不训练 MemAdaptor。
# 本脚本显式打开：① reward_model.format_reward（think/action/memory_retrieve shaping）
# ② env.memory.experience_utility（c_use/c_succ Laplace 写回 + 可选剪枝）。
# 需本地 vLLM rollout（不可使用 openai_api 作为主 policy——无法正确反传 / logprob）。
# 论文框架见 https://arxiv.org/abs/2510.16079
set -x
set -euo pipefail
export RAY_ADDRESS='http://10.140.37.38:8265'

ENGINE="vllm"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES 2>/dev/null || true
export HYDRA_FULL_ERROR=1
export WANDB_MODE="offline"

trainer_n_gpus_per_node=4
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
# full=多字段 JSON（适合强模型/teacher）；compact=只让模型写 memory_text，state/action 从轨迹回填（适合小模型自蒸馏）
EXPERIENCE_SUMMARIZER_SCHEMA="compact"
RETRIEVAL_MODE="agentic" # agentic | fixed（EvolveR 在线阶段常用 agentic 检索）
RETRIEVE_KEY="memory_text"
EMBEDDING_API_URL="http://10.140.37.18:8887/v1"
EMBEDDING_API_KEY="DataFrontier_bge_m3"

MEMORY_REMOTE_SLURM=True
MEMORY_REMOTE_PARTITION="DataFrontier_Explore"
MEMORY_REMOTE_SERVER_PORT="8765"
MEMORY_REMOTE_EXCLUDE_NODES="SH-IDC1-10-140-37-11"
MEMORY_APPTAINER_SIF="/mnt/petrelfs/wurong/glibc_ubuntu22.sif"
MEMORY_CONDA_SH="/mnt/petrelfs/wurong/miniconda3/etc/profile.d/conda.sh"
MEMORY_REMOTE_CONDA_ENV="verl-agent"

# MEMORY_REBUILD_SOURCE_PATH="data/MemAdaptor/AgentTraj-L/${TASK_NAME}_train_memory_records-gpt-5.1.jsonl"
MEMORY_REBUILD_SOURCE_PATH=""

# --- reward_model.format_reward：与 projection 对齐的 think / action / memory_retrieve shaping ---
# 见 verl/trainer/config/ppo_trainer.yaml ``reward_model.format_reward``；此处用 Hydra 覆盖。
FORMAT_REWARD_ENABLE=True
FORMAT_WEIGHT_OUTCOME=1.0
FORMAT_WEIGHT_FORMAT=0.1
# agentic 检索时建议 True，要求响应里出现成对 memory 检索标签（与 env.memory.retrieval_query_* 一致）。
FORMAT_REQUIRE_MEMORY_RETRIEVE=True

# --- env.memory.experience_utility：EvolveR 式 c_use/c_succ + Laplace 写回 value，可选剪枝 ---
# prune_every_n_global_steps=0 表示不剪枝；设为正整数则每 N 个 trainer step 剪一次低分记忆。
EXPERIENCE_UTILITY_ENABLE=True
EXPERIENCE_UTILITY_UPDATE_ON_RETRIEVAL=True
EXPERIENCE_UTILITY_UPDATE_ON_EPISODE_END=True
EXPERIENCE_UTILITY_PRUNE_EVERY_N_GLOBAL_STEPS=20
EXPERIENCE_UTILITY_PRUNE_SCORE_THRESHOLD=0.3
EXPERIENCE_UTILITY_MIN_USES_BEFORE_PRUNE=3

EXPERIMENT_NAME="train_evolver-1.5B"
EXPERIMENTS_ROOT="data/MemAdaptor/exp_results"
MODEL_PATH="models/public_models/Qwen2.5-1.5B-Instruct"

if [ "${MEMORY_ENABLED}" = "True" ]; then
  EXPERIMENT_NAME="${EXPERIMENT_NAME}-with_${RETRIEVAL_MODE}_memory"
  EXPERIMENT_NAME="${EXPERIMENT_NAME}-retrieve_${RETRIEVE_KEY}"
  EXPERIMENT_NAME="${EXPERIMENT_NAME}-${EXPERIENCE_SUMMARIZER_MODE}_distill"
else
  EXPERIMENT_NAME="${EXPERIMENT_NAME}-no_memory"
fi

EXP_DIR="${EXPERIMENTS_ROOT}/${TASK_NAME}/${EXPERIMENT_NAME}"
MEMORY_STORE_DIR="${EXP_DIR}/memory_vdb"
TRAINER_CHECKPOINT_DIR="models/save_models/mem_adaptor/${EXPERIMENT_NAME}"

mkdir -p "${EXP_DIR}"
mkdir -p "${TRAINER_CHECKPOINT_DIR}"
LOG_FILE="${EXP_DIR}/train_alfworld_evolver-$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "${LOG_FILE}") 2>&1
echo "[log] Writing full run output to: ${LOG_FILE}"
echo "[log] trainer.default_local_dir=${TRAINER_CHECKPOINT_DIR}"

# 训练 batch：须与 env.rollout.n（GRPO group）及数据量匹配
train_data_size=16
val_data_size=140  ## alfworld验证集只有140条数据，需要整除val_batch_size
group_size=8

# 多轮只认 data.max_prompt_length；经验写回 summarizer 需要更大 prompt 预算时，必须同时抬高 vLLM max_model_len
# （否则 summarizer 会被 clamp 到 max_model_len - response_length，见 experience_summarizer 警告）。
DATA_MAX_PROMPT_LENGTH="${DATA_MAX_PROMPT_LENGTH:-2048}"
DATA_MAX_RESPONSE_LENGTH="${DATA_MAX_RESPONSE_LENGTH:-512}"
SUMMARIZER_MAX_PROMPT_TOKENS="${SUMMARIZER_MAX_PROMPT_TOKENS:-12288}"
ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-}"
if [ -z "${ROLLOUT_MAX_MODEL_LEN}" ]; then
  ROLLOUT_MAX_MODEL_LEN=$((SUMMARIZER_MAX_PROMPT_TOKENS + DATA_MAX_RESPONSE_LENGTH))
fi

# 与 trainer_n_gpus_per_node、模型宽度匹配（单卡可设 1）
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

FORMAT_REWARD_CLI=(
  reward_model.format_reward.enable="${FORMAT_REWARD_ENABLE}"
  reward_model.format_reward.weight_outcome="${FORMAT_WEIGHT_OUTCOME}"
  reward_model.format_reward.weight_format="${FORMAT_WEIGHT_FORMAT}"
  reward_model.format_reward.require_memory_retrieve="${FORMAT_REQUIRE_MEMORY_RETRIEVE}"
  reward_model.format_reward.format_warmup_global_steps=30
  reward_model.format_reward.warmup_weight_format_multiplier=0.0
  reward_model.format_reward.warmup_require_memory_retrieve=False
  reward_model.format_reward.warmup_penalize_chinese_chars=True
)

EXPERIENCE_UTILITY_CLI=()
if [ "${MEMORY_ENABLED}" = "True" ]; then
  EXPERIENCE_UTILITY_CLI=(
    env.memory.experience_utility.enable="${EXPERIENCE_UTILITY_ENABLE}"
    env.memory.experience_utility.update_on_retrieval="${EXPERIENCE_UTILITY_UPDATE_ON_RETRIEVAL}"
    env.memory.experience_utility.update_on_episode_end="${EXPERIENCE_UTILITY_UPDATE_ON_EPISODE_END}"
    env.memory.experience_utility.prune_every_n_global_steps="${EXPERIENCE_UTILITY_PRUNE_EVERY_N_GLOBAL_STEPS}"
    env.memory.experience_utility.prune_score_threshold="${EXPERIENCE_UTILITY_PRUNE_SCORE_THRESHOLD}"
    env.memory.experience_utility.min_uses_before_prune="${EXPERIENCE_UTILITY_MIN_USES_BEFORE_PRUNE}"
  )
else
  EXPERIENCE_UTILITY_CLI=(env.memory.experience_utility.enable=False)
fi

ray job submit --runtime-env-json "${RAY_JOB_RUNTIME_ENV_JSON}" -- \
    python3 -m verl.trainer.main_ppo \
      algorithm.adv_estimator=grpo \
      data.train_files="${TRAIN_FILE}" \
      data.val_files="${VAL_FILE}" \
      data.train_batch_size="${train_data_size}" \
      data.val_batch_size="${val_data_size}" \
      data.max_prompt_length="${DATA_MAX_PROMPT_LENGTH}" \
      data.max_response_length="${DATA_MAX_RESPONSE_LENGTH}" \
      data.filter_overlong_prompts=True \
      data.truncation='error' \
      data.return_raw_chat=True \
      actor_rollout_ref.model.path="${MODEL_PATH}" \
      actor_rollout_ref.actor.trainable=true \
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
      actor_rollout_ref.rollout.max_model_len="${ROLLOUT_MAX_MODEL_LEN}" \
      actor_rollout_ref.rollout.name="${ENGINE}" \
      actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
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
      env.memory.enabled="${MEMORY_ENABLED}" \
      env.memory.store_dir="${MEMORY_STORE_DIR}" \
      env.memory.write_back="${MEMORY_WRITE_BACK}" \
      env.memory.experience_summarizer.mode="${EXPERIENCE_SUMMARIZER_MODE}" \
      env.memory.experience_summarizer.schema="${EXPERIENCE_SUMMARIZER_SCHEMA}" \
      env.memory.experience_summarizer.summarizer_max_prompt_tokens="${SUMMARIZER_MAX_PROMPT_TOKENS}" \
      env.memory.retrieval_mode="${RETRIEVAL_MODE}" \
      env.memory.retrieve_key="${RETRIEVE_KEY}" \
      "${EXPERIENCE_UTILITY_CLI[@]+"${EXPERIENCE_UTILITY_CLI[@]}"}" \
      "${MEMORY_CLI[@]+"${MEMORY_CLI[@]}"}" \
      "${REMOTE_VDB_CLI[@]+"${REMOTE_VDB_CLI[@]}"}" \
      "${FORMAT_REWARD_CLI[@]+"${FORMAT_REWARD_CLI[@]}"}" \
      env.rollout.n="${group_size}" \
      env.resources_per_worker.num_cpus="${num_cpus_per_env_worker}" \
      trainer.critic_warmup=0 \
      trainer.logger=['console','wandb'] \
      trainer.project_name='MemAdaptor_alfworld' \
      trainer.experiment_name="${EXPERIMENT_NAME}" \
      trainer.default_local_dir="${TRAINER_CHECKPOINT_DIR}" \
      trainer.n_gpus_per_node="${trainer_n_gpus_per_node}" \
      trainer.nnodes=1 \
      trainer.save_freq=30 \
      trainer.test_freq=5 \
      trainer.total_epochs=150 \
      trainer.validation_data_dir="${EXP_DIR}/val_traj" \
      trainer.val_before_train=False \
      trainer.val_only=False \
      "$@"