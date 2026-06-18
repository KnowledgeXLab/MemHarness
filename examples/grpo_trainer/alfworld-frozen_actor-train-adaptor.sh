#!/usr/bin/env bash
#SBATCH --job-name=e05-frozen-qwen2.5_7b-train_adaptor_3B
#SBATCH --partition=DataFrontier_Explore
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=500G 
#SBATCH --output=logs/mem_adaptor/alfworld/frozen_qwen2.5_7b-train_adaptor_3B_%j.out
#SBATCH --error=logs/mem_adaptor/alfworld/frozen_qwen2.5_7b-train_adaptor_3B_%j.err

set -x
set -euo pipefail

mkdir -p logs/mem_adaptor/alfworld

ENGINE="vllm"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES 2>/dev/null || true
export HYDRA_FULL_ERROR=1
export WANDB_MODE="offline"

REASONING_MODEL_PATH="models/public_models/Qwen2.5-7B-Instruct"
# REASONING_MODEL_PATH='models/save_models/mem_adaptor/cold_start/alfworld/qwen2.5-7b-cold-start-20260430/global_step_125'
# REASONING_MODEL_PATH='models/save_models/mem_adaptor/cold_start/alfworld/qwen2.5-1.5b-cold-start-20260519/global_step_250'
# MemAdaptor 专用池上的模型（可与 Reasoning 相同或更小）
MEM_ADAPTOR_MODEL_PATH="models/public_models/Qwen2.5-3B-Instruct"
# 可选：单独指定 KL ref；留空则默认与 MEM_ADAPTOR_MODEL_PATH 相同（worker 初始化时冻结）
MEM_ADAPTOR_REF_MODEL_PATH="${MEM_ADAPTOR_REF_MODEL_PATH:-}"

# --- 与 train_alfworld-adaptor-same 一致：可选按 global_step 切换检索 / Adaptor env 步调度 ---
MEM_ADAPTOR_USE_RECOMMENDED_PHASES="0"

# global_pool：Reasoning（vLLM+FSDP actor 等）每节点 GPU 数；须 >= REASONING_TENSOR_PARALLEL_SIZE
trainer_n_gpus_per_node=4
# mem_adaptor 专用池每节点 GPU 数（8 卡节点上 trainer_n + mem_adaptor 应 <= 8）
mem_adaptor_gpus_per_node=4

# --- Adaptor GRPO reward 系数（可用环境变量覆盖，例如 MEM_ADAPTOR_GRPO_EPISODE_RETURN_COEF=0.3 sbatch ...）---
# score = episode_return_coef * R_episode
#       + step_reward_coef * max(0, R_next - R_at_adaptor)     [step_reward_enable]
#       - identical_coef * identical_penalty                   [identical_enable]
#       - english_coef * english_penalty                       [english_enable]
MEM_ADAPTOR_GRPO_EPISODE_RETURN_COEF="${MEM_ADAPTOR_GRPO_EPISODE_RETURN_COEF:-1.0}"
MEM_ADAPTOR_GRPO_STEP_REWARD_ENABLE="${MEM_ADAPTOR_GRPO_STEP_REWARD_ENABLE:-true}"
MEM_ADAPTOR_GRPO_STEP_REWARD_COEF="${MEM_ADAPTOR_GRPO_STEP_REWARD_COEF:-1.0}"
MEM_ADAPTOR_GRPO_IDENTICAL_ENABLE="${MEM_ADAPTOR_GRPO_IDENTICAL_ENABLE:-true}"
MEM_ADAPTOR_GRPO_IDENTICAL_COEF="${MEM_ADAPTOR_GRPO_IDENTICAL_COEF:-1.0}"
MEM_ADAPTOR_GRPO_IDENTICAL_PENALTY="${MEM_ADAPTOR_GRPO_IDENTICAL_PENALTY:-2.0}"
MEM_ADAPTOR_GRPO_ENGLISH_ENABLE="${MEM_ADAPTOR_GRPO_ENGLISH_ENABLE:-true}"
MEM_ADAPTOR_GRPO_ENGLISH_COEF="${MEM_ADAPTOR_GRPO_ENGLISH_COEF:-1.0}"
MEM_ADAPTOR_GRPO_ENGLISH_PENALTY="${MEM_ADAPTOR_GRPO_ENGLISH_PENALTY:-1.0}"
# 同一条轨迹多次 adaptor 调用时，GRPO advantage 再除以调用次数 K
MEM_ADAPTOR_GRPO_DIVIDE_ADV_BY_TRAJ_STEPS="${MEM_ADAPTOR_GRPO_DIVIDE_ADV_BY_TRAJ_STEPS:-true}"


if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  # sbatch 会把脚本复制到 /var/spool/slurmd/job*/slurm_script，BASH_SOURCE 不是仓库路径
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
  SCRIPT_DIR="${REPO_ROOT}/examples/grpo_trainer"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
cd "${REPO_ROOT}"
export MEMADAPTOR_REPO_ROOT="${MEMADAPTOR_REPO_ROOT:-${REPO_ROOT}}"

# shellcheck source=memory_eval_helpers.sh
if [[ ! -f "${SCRIPT_DIR}/memory_eval_helpers.sh" ]]; then
  echo "[error] memory_eval_helpers.sh not found: ${SCRIPT_DIR}/memory_eval_helpers.sh" >&2
  echo "[error] Submit from repo root: sbatch examples/grpo_trainer/alfworld-frozen_actor-train-adaptor.sh" >&2
  exit 1
fi
source "${SCRIPT_DIR}/memory_eval_helpers.sh"

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
RETRIEVAL_MODE="agentic" # agentic | fixed（与 remote 默认一致）
RETRIEVE_KEY="memory_text" # memory_text | state_text
# agentic 评测通用 ckpt 时可设 RETRIEVAL_MODE=agentic 并启用下方检索引导（见 memory_eval_helpers.sh）
USE_GENERAL_MODEL_RETRIEVAL_HINT="${USE_GENERAL_MODEL_RETRIEVAL_HINT:-1}"
RETRIEVAL_INSTRUCTION_PROMPT="${RETRIEVAL_INSTRUCTION_PROMPT:-}"
RETRIEVAL_INSTRUCTION_PROMPT_FILE="${RETRIEVAL_INSTRUCTION_PROMPT_FILE:-}"
# EMBEDDING_API_URL="http://10.140.37.18:8887/v1"
# EMBEDDING_API_KEY="DataFrontier_bge_m3"
EMBEDDING_API_URL="http://10.140.37.57:8081/v1"
EMBEDDING_API_KEY="DataFrontier_bge_m3"

MEMORY_REMOTE_SLURM=True
MEMORY_REMOTE_PARTITION="DataFrontier_Explore"  # DataFrontier_Explore / p-cpu-new
MEMORY_REMOTE_SERVER_PORT="8765"
# 远程起 VDB 的 sbatch：Slurm --exclude，逗号分隔；Hydra 需列表语法，见下方 mem_exclude_to_hydra_list
MEMORY_REMOTE_EXCLUDE_NODES=''
MEMORY_APPTAINER_SIF="/mnt/petrelfs/wurong/glibc_ubuntu22.sif"
MEMORY_CONDA_SH="/mnt/petrelfs/wurong/miniconda3/etc/profile.d/conda.sh"
MEMORY_REMOTE_CONDA_ENV="verl-agent"

# MEMORY_REBUILD_SOURCE_PATH="data/MemAdaptor/AgentTraj-L/${TASK_NAME}_train_memory_records-gpt-5.1.jsonl"
MEMORY_REBUILD_SOURCE_PATH=""

EXPERIENCE_UTILITY_ENABLE=True
EXPERIENCE_UTILITY_PRUNE_EVERY_N_GLOBAL_STEPS=20
EXPERIENCE_UTILITY_PRUNE_SCORE_THRESHOLD=0.3
EXPERIENCE_UTILITY_MIN_USES_BEFORE_PRUNE=3

EXPERIMENT_NAME="frozen_qwen2.5_7b-train_adaptor_3B"
EXPERIMENTS_ROOT="data/MemAdaptor/exp_results"

if [ "${MEMORY_ENABLED}" = "True" ]; then
  EXPERIMENT_NAME="${EXPERIMENT_NAME}-with_${RETRIEVAL_MODE}_memory"
  EXPERIMENT_NAME="${EXPERIMENT_NAME}-retrieve_${RETRIEVE_KEY}"
  EXPERIMENT_NAME="${EXPERIMENT_NAME}-${EXPERIENCE_SUMMARIZER_MODE}_distill"
else
  EXPERIMENT_NAME="${EXPERIMENT_NAME}-no_memory"
fi

EXP_DIR="${EXPERIMENTS_ROOT}/${TASK_NAME}/${EXPERIMENT_NAME}"
MEMORY_STORE_DIR="${EXP_DIR}/memory_vdb"
TRAINER_CHECKPOINT_DIR="models/save_models/mem_adaptor/${TASK_NAME}/${EXPERIMENT_NAME}"

mkdir -p "${EXP_DIR}"
mkdir -p "${TRAINER_CHECKPOINT_DIR}"
LOG_FILE="${EXP_DIR}/train_alfworld_adaptor_local_new-$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "${LOG_FILE}") 2>&1
echo "[log] Writing full run output to: ${LOG_FILE}"
echo "[log] REASONING_MODEL_PATH=${REASONING_MODEL_PATH}"
echo "[log] MEM_ADAPTOR_MODEL_PATH=${MEM_ADAPTOR_MODEL_PATH}"
if [ -n "${MEM_ADAPTOR_REF_MODEL_PATH}" ]; then
  echo "[log] MEM_ADAPTOR_REF_MODEL_PATH=${MEM_ADAPTOR_REF_MODEL_PATH}"
else
  echo "[log] MEM_ADAPTOR_REF_MODEL_PATH=(unset, KL ref uses MEM_ADAPTOR_MODEL_PATH)"
fi
echo "[log] MEM_ADAPTOR_GRPO_EPISODE_RETURN_COEF=${MEM_ADAPTOR_GRPO_EPISODE_RETURN_COEF}"
echo "[log] MEM_ADAPTOR_GRPO_STEP_REWARD_ENABLE=${MEM_ADAPTOR_GRPO_STEP_REWARD_ENABLE} coef=${MEM_ADAPTOR_GRPO_STEP_REWARD_COEF}"
echo "[log] MEM_ADAPTOR_GRPO_IDENTICAL_ENABLE=${MEM_ADAPTOR_GRPO_IDENTICAL_ENABLE} coef=${MEM_ADAPTOR_GRPO_IDENTICAL_COEF} penalty=${MEM_ADAPTOR_GRPO_IDENTICAL_PENALTY}"
echo "[log] MEM_ADAPTOR_GRPO_ENGLISH_ENABLE=${MEM_ADAPTOR_GRPO_ENGLISH_ENABLE} coef=${MEM_ADAPTOR_GRPO_ENGLISH_COEF} penalty=${MEM_ADAPTOR_GRPO_ENGLISH_PENALTY}"
echo "[log] MEM_ADAPTOR_GRPO_DIVIDE_ADV_BY_TRAJ_STEPS=${MEM_ADAPTOR_GRPO_DIVIDE_ADV_BY_TRAJ_STEPS}"
echo "[log] trainer.default_local_dir=${TRAINER_CHECKPOINT_DIR}"

# 训练 batch：须与 env.rollout.n（GRPO group）及数据量匹配
train_data_size=16
val_data_size=140  ## alfworld验证集只有140条数据，需要整除val_batch_size
group_size=8

# 多轮只认 data.max_prompt_length；经验写回 summarizer 需要更大 prompt 预算时，必须同时抬高 vLLM max_model_len
# （否则 summarizer 会被 clamp 到 max_model_len - response_length，见 experience_summarizer 警告）。  
DATA_MAX_PROMPT_LENGTH=2048
DATA_MAX_RESPONSE_LENGTH=512
SUMMARIZER_MAX_PROMPT_TOKENS=12288
ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-}"
if [ -z "${ROLLOUT_MAX_MODEL_LEN}" ]; then
  ROLLOUT_MAX_MODEL_LEN=$((SUMMARIZER_MAX_PROMPT_TOKENS + DATA_MAX_RESPONSE_LENGTH))
fi

# vLLM TP；须与 Reasoning 池 GPU 数一致且 trainer_n_gpus_per_node 可被整除
tensor_model_parallel_size=2
if (( trainer_n_gpus_per_node % tensor_model_parallel_size != 0 )); then
  echo "[error] trainer_n_gpus_per_node=${trainer_n_gpus_per_node} must be divisible by REASONING_TENSOR_PARALLEL_SIZE=${tensor_model_parallel_size}" >&2
  exit 1
fi

VALIDATE_ON_TRAIN_SPLIT=False

PREPARE_FLAGS=()
if [ "${PREPARE_OVERWRITE:-0}" = "1" ] || [ "${PREPARE_OVERWRITE:-}" = "true" ]; then
  PREPARE_FLAGS+=(--overwrite)
fi

python3 -m examples.data_preprocess.prepare \
  --mode 'text' \
  --local_dir "${DATA_ROOT}" \
  --infer_alfworld_sizes \
  --overwrite \
  --alfworld_eval_split eval_in_distribution \
  "${PREPARE_FLAGS[@]+"${PREPARE_FLAGS[@]}"}"

MEMORY_CLI=()
# ALFWorld 任务对环境细节非常敏感，关闭 dedupe 防止相似但不同的记忆被错误去重
MEMORY_CLI+=(env.memory.memory_text_retrieval_dedupe_similarity_threshold=null)
MEMORY_CLI+=(env.memory.memory_text_insert_dedupe_similarity_threshold=null)
if [ -n "${MEMORY_REBUILD_SOURCE_PATH}" ]; then
  MEMORY_CLI+=(env.memory.rebuild_source_path="${MEMORY_REBUILD_SOURCE_PATH}")
fi
if [ -n "${EMBEDDING_API_URL}" ]; then
  MEMORY_CLI+=(env.memory.embedding_api_url="${EMBEDDING_API_URL}")
fi
if [ -n "${EMBEDDING_API_KEY}" ]; then
  MEMORY_CLI+=(env.memory.embedding_api_key="${EMBEDDING_API_KEY}")
fi

append_retrieval_instruction_cli

export VLLM_NCCL_SO_PATH=/mnt/petrelfs/wurong/miniconda3/envs/verl-agent/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2

# Slurm --exclude 逗号分隔多节点。Hydra 对 ``a,b`` 会报 Ambiguous；用列表语法 ``['a','b']``
mem_exclude_to_hydra_list() {
  local s="${1:-}" IFS=,
  read -r -a _ex_parts <<< "$s" || true
  local out="[" first=1
  for n in "${_ex_parts[@]}"; do
    n="${n#"${n%%[![:space:]]*}"}"
    n="${n%"${n##*[![:space:]]}"}"
    [ -z "$n" ] && continue
    if [ "$first" -eq 0 ]; then out+=","; fi
    out+="'${n//\'/\\\'}'"
    first=0
  done
  out+=']'
  printf '%s' "$out"
}

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
    _ex_hy="$(mem_exclude_to_hydra_list "${MEMORY_REMOTE_EXCLUDE_NODES}")"
    REMOTE_VDB_CLI+=("env.memory.remote_slurm_launch.exclude_nodes=${_ex_hy}")
  fi
fi

FORMAT_REWARD_CLI=(
  reward_model.format_reward.enable=True
  reward_model.format_reward.weight_outcome=1.0
  reward_model.format_reward.weight_format=0.1
  reward_model.format_reward.require_memory_retrieve=True
  reward_model.format_reward.format_warmup_global_steps=0
  reward_model.format_reward.warmup_weight_format_multiplier=0.0
  reward_model.format_reward.warmup_require_memory_retrieve=False
  reward_model.format_reward.warmup_penalize_chinese_chars=False
)

EXPERIENCE_UTILITY_CLI=()
if [ "${MEMORY_ENABLED}" = "True" ]; then
  EXPERIENCE_UTILITY_CLI=(
    env.memory.experience_utility.enable="${EXPERIENCE_UTILITY_ENABLE}"
    env.memory.experience_utility.prune_every_n_global_steps="${EXPERIENCE_UTILITY_PRUNE_EVERY_N_GLOBAL_STEPS}"
    env.memory.experience_utility.prune_score_threshold="${EXPERIENCE_UTILITY_PRUNE_SCORE_THRESHOLD}"
    env.memory.experience_utility.min_uses_before_prune="${EXPERIENCE_UTILITY_MIN_USES_BEFORE_PRUNE}"
  )
else
  EXPERIENCE_UTILITY_CLI=(env.memory.experience_utility.enable=False)
fi

MEM_ADAPTOR_PHASES_CLI=()
if [ "${MEM_ADAPTOR_USE_RECOMMENDED_PHASES}" = "1" ]; then
  MEM_ADAPTOR_PHASES_CLI+=(
    'env.memory.retrieval_mode_phases=[{global_step_start: 0, global_step_end: 50, mode: fixed}, {global_step_start: 50, global_step_end: null, mode: agentic}]'
    'mem_adaptor.env_step_phases=[{global_step_start: 0, global_step_end: 50, env_step_start: 1, env_step_end: 51, env_step_every_n: 1}, {global_step_start: 51, global_step_end: null, env_step_start: null, env_step_end: null, env_step_every_n: 1}]'
  )
fi

MEM_ADAPTOR_REF_CLI=()
if [ -n "${MEM_ADAPTOR_REF_MODEL_PATH}" ]; then
  MEM_ADAPTOR_REF_CLI+=(mem_adaptor.ref_model.path="${MEM_ADAPTOR_REF_MODEL_PATH}")
fi

MEM_ADAPTOR_REWARD_CLI=(
  mem_adaptor.grpo_reward.episode_return_coef="${MEM_ADAPTOR_GRPO_EPISODE_RETURN_COEF}"
  mem_adaptor.grpo_step_reward_shaping.enable="${MEM_ADAPTOR_GRPO_STEP_REWARD_ENABLE}"
  mem_adaptor.grpo_step_reward_shaping.coef="${MEM_ADAPTOR_GRPO_STEP_REWARD_COEF}"
  mem_adaptor.grpo_identical_rewrite_penalty.enable="${MEM_ADAPTOR_GRPO_IDENTICAL_ENABLE}"
  mem_adaptor.grpo_identical_rewrite_penalty.coef="${MEM_ADAPTOR_GRPO_IDENTICAL_COEF}"
  mem_adaptor.grpo_identical_rewrite_penalty.penalty="${MEM_ADAPTOR_GRPO_IDENTICAL_PENALTY}"
  mem_adaptor.grpo_english_shaping.enable="${MEM_ADAPTOR_GRPO_ENGLISH_ENABLE}"
  mem_adaptor.grpo_english_shaping.coef="${MEM_ADAPTOR_GRPO_ENGLISH_COEF}"
  mem_adaptor.grpo_english_shaping.penalty="${MEM_ADAPTOR_GRPO_ENGLISH_PENALTY}"
  mem_adaptor.grpo_divide_advantage_by_traj_adaptor_steps="${MEM_ADAPTOR_GRPO_DIVIDE_ADV_BY_TRAJ_STEPS}"
)

unset RAY_ADDRESS
ray stop --force || true
ray start --head
sleep 5

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
  actor_rollout_ref.model.path="${REASONING_MODEL_PATH}" \
  actor_rollout_ref.actor.trainable=false \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=256 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.model.enable_gradient_checkpointing=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${tensor_model_parallel_size}" \
  actor_rollout_ref.rollout.max_model_len="${ROLLOUT_MAX_MODEL_LEN}" \
  actor_rollout_ref.rollout.name="${ENGINE}" \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
  actor_rollout_ref.rollout.enable_chunked_prefill=true \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.use_invalid_action_penalty=True \
  actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
  algorithm.use_kl_in_reward=False \
  mem_adaptor.enable=true \
  mem_adaptor.use_actor_rollout_wg=false \
  mem_adaptor.train_memory_adaptor=true \
  mem_adaptor.model.path="${MEM_ADAPTOR_MODEL_PATH}" \
  "${MEM_ADAPTOR_REF_CLI[@]+"${MEM_ADAPTOR_REF_CLI[@]}"}" \
  mem_adaptor.actor_use_kl_loss=true \
  mem_adaptor.actor_kl_loss_coef=0.01 \
  mem_adaptor.actor_kl_loss_type=low_var_kl \
  mem_adaptor.ref_param_offload=true \
  mem_adaptor.resource_pool_gpus_per_node="[${mem_adaptor_gpus_per_node}]" \
  mem_adaptor.max_new_tokens=128 \
  "${MEM_ADAPTOR_REWARD_CLI[@]}" \
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
  "${MEM_ADAPTOR_PHASES_CLI[@]+"${MEM_ADAPTOR_PHASES_CLI[@]}"}" \
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
  trainer.save_freq=50 \
  trainer.save_best_val_ckpt=True \
  trainer.save_best_val_mode="max" \
  trainer.save_best_val_metric="val/success_rate" \
  trainer.test_freq=5 \
  trainer.total_epochs=150 \
  trainer.validation_data_dir="${EXP_DIR}/val_traj" \
  trainer.val_before_train=False \
  trainer.val_only=False \
  "$@"
