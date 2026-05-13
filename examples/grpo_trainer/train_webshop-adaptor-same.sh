#!/usr/bin/env bash
# MemAdaptor + WebShop：与 train_alfworld-adaptor-same.sh 对齐的「同一套 Actor/vLLM」配方：
#   MODEL_PATH 同时作为 actor_rollout_ref.model.path 与 mem_adaptor.model.path，
#   mem_adaptor.use_actor_rollout_wg=true（Adaptor 与主 policy 共用 rollout，不建 mem_adaptor_pool），
#   mem_adaptor.train_memory_adaptor=false。
# 数据与环境默认值对齐 train_webshop-adaptor-local.sh / run_webshop.sh。
#
# 用法：
#   ./train_webshop-adaptor-same.sh           # 默认 ENGINE=vllm
#   ./train_webshop-adaptor-same.sh hf
set -x
set -euo pipefail

ENGINE="${1:-vllm}"

export RAY_ADDRESS="${RAY_ADDRESS:-http://10.140.37.15:8265}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES 2>/dev/null || true
export HYDRA_FULL_ERROR=1
export WANDB_MODE="${WANDB_MODE:-offline}"

# 单一 checkpoint：同时作为 actor_rollout_ref.model.path 与 mem_adaptor.model.path
MODEL_PATH="${MODEL_PATH:-models/public_models/Qwen2.5-3B-Instruct}"

MEM_ADAPTOR_USE_RECOMMENDED_PHASES="${MEM_ADAPTOR_USE_RECOMMENDED_PHASES:-0}"

# global_pool：Reasoning（vLLM+FSDP actor/ref 等）每节点 GPU 数。
# use_actor_rollout_wg=true 时不占用 mem_adaptor.resource_pool_gpus_per_node。
trainer_n_gpus_per_node="${trainer_n_gpus_per_node:-4}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export MEMADAPTOR_REPO_ROOT="${MEMADAPTOR_REPO_ROOT:-${REPO_ROOT}}"

DATA_ROOT="${DATA_ROOT:-data/verl-agent}"
TRAIN_FILE="${DATA_ROOT}/text/train.parquet"
TEST_FILE="${DATA_ROOT}/text/test.parquet"
VAL_FILE="${TEST_FILE}"

WEBSHOP_DATA_DIR="${WEBSHOP_DATA_DIR:-}"
WEBSHOP_USE_SMALL="${WEBSHOP_USE_SMALL:-1}"
WEBSHOP_TRAIN_CLI=()
if [ -n "${WEBSHOP_DATA_DIR}" ]; then
  WEBSHOP_TRAIN_CLI+=(env.webshop.data_dir="${WEBSHOP_DATA_DIR}")
fi
if [ "${WEBSHOP_USE_SMALL}" = "0" ] || [ "${WEBSHOP_USE_SMALL}" = "false" ]; then
  WEBSHOP_TRAIN_CLI+=(env.webshop.use_small=false)
fi

export WANDB_DIR="${WANDB_DIR:-wandb_logs}"

num_cpus_per_env_worker="${num_cpus_per_env_worker:-0.1}"

TASK_NAME="webshop"
ENV_NAME="${ENV_NAME:-Webshop}"

MEMORY_ENABLED="${MEMORY_ENABLED:-True}"
MEMORY_WRITE_BACK="${MEMORY_WRITE_BACK:-True}"
EXPERIENCE_SUMMARIZER_MODE="${EXPERIENCE_SUMMARIZER_MODE:-self}"
EXPERIENCE_SUMMARIZER_SCHEMA="${EXPERIENCE_SUMMARIZER_SCHEMA:-compact}"
RETRIEVAL_MODE="${RETRIEVAL_MODE:-agentic}"
RETRIEVE_KEY="${RETRIEVE_KEY:-memory_text}"

EMBEDDING_API_URL="${EMBEDDING_API_URL:-http://10.140.37.140:8081/v1}"
EMBEDDING_API_KEY="${EMBEDDING_API_KEY:-}"

MEMORY_REMOTE_SLURM="${MEMORY_REMOTE_SLURM:-True}"
MEMORY_REMOTE_PARTITION="${MEMORY_REMOTE_PARTITION:-DataFrontier_Explore}"
MEMORY_REMOTE_SERVER_PORT="${MEMORY_REMOTE_SERVER_PORT:-8765}"
MEMORY_REMOTE_EXCLUDE_NODES="${MEMORY_REMOTE_EXCLUDE_NODES:-}"
MEMORY_APPTAINER_SIF="${MEMORY_APPTAINER_SIF:-/mnt/petrelfs/wurong/glibc_ubuntu22.sif}"
MEMORY_CONDA_SH="${MEMORY_CONDA_SH:-/mnt/petrelfs/wurong/miniconda3/etc/profile.d/conda.sh}"
MEMORY_REMOTE_CONDA_ENV="${MEMORY_REMOTE_CONDA_ENV:-verl-agent}"

MEMORY_REBUILD_SOURCE_PATH="${MEMORY_REBUILD_SOURCE_PATH:-}"

EXPERIENCE_UTILITY_ENABLE="${EXPERIENCE_UTILITY_ENABLE:-True}"
EXPERIENCE_UTILITY_PRUNE_EVERY_N_GLOBAL_STEPS="${EXPERIENCE_UTILITY_PRUNE_EVERY_N_GLOBAL_STEPS:-20}"
EXPERIENCE_UTILITY_PRUNE_SCORE_THRESHOLD="${EXPERIENCE_UTILITY_PRUNE_SCORE_THRESHOLD:-0.3}"
EXPERIENCE_UTILITY_MIN_USES_BEFORE_PRUNE="${EXPERIENCE_UTILITY_MIN_USES_BEFORE_PRUNE:-3}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-webshop_train_adaptor-same-3b}"
EXPERIMENTS_ROOT="${EXPERIMENTS_ROOT:-data/MemAdaptor/exp_results}"

if [ "${MEMORY_ENABLED}" = "True" ]; then
  EXPERIMENT_NAME="${EXPERIMENT_NAME}-with_${RETRIEVAL_MODE}_memory"
  EXPERIMENT_NAME="${EXPERIMENT_NAME}-retrieve_${RETRIEVE_KEY}"
  EXPERIMENT_NAME="${EXPERIMENT_NAME}-${EXPERIENCE_SUMMARIZER_MODE}_distill"
else
  EXPERIMENT_NAME="${EXPERIMENT_NAME}-no_memory"
fi

EXP_DIR="${EXPERIMENTS_ROOT}/${TASK_NAME}/${EXPERIMENT_NAME}"
MEMORY_STORE_DIR="${EXP_DIR}/memory_vdb"
TRAINER_CHECKPOINT_DIR="${TRAINER_CHECKPOINT_DIR:-models/save_models/mem_adaptor/${EXPERIMENT_NAME}}"

mkdir -p "${EXP_DIR}"
mkdir -p "${TRAINER_CHECKPOINT_DIR}"
LOG_FILE="${EXP_DIR}/train_webshop_adaptor_same-$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "${LOG_FILE}") 2>&1
echo "[log] Writing full run output to: ${LOG_FILE}"
echo "[log] ENV_NAME=${ENV_NAME}"
echo "[log] ENGINE=${ENGINE}"
echo "[log] MODEL_PATH (Reasoning + MemAdaptor)=${MODEL_PATH}"
echo "[log] trainer.default_local_dir=${TRAINER_CHECKPOINT_DIR}"

train_data_size="${train_data_size:-16}"
val_data_size="${val_data_size:-128}"
group_size="${group_size:-8}"

DATA_MAX_PROMPT_LENGTH="${DATA_MAX_PROMPT_LENGTH:-4096}"
DATA_MAX_RESPONSE_LENGTH="${DATA_MAX_RESPONSE_LENGTH:-512}"
SUMMARIZER_MAX_PROMPT_TOKENS="${SUMMARIZER_MAX_PROMPT_TOKENS:-12288}"
ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-}"
if [ -z "${ROLLOUT_MAX_MODEL_LEN}" ]; then
  ROLLOUT_MAX_MODEL_LEN=$((SUMMARIZER_MAX_PROMPT_TOKENS + DATA_MAX_RESPONSE_LENGTH))
fi

tensor_model_parallel_size="${tensor_model_parallel_size:-2}"

PREPARE_FLAGS=()
if [ "${PREPARE_OVERWRITE:-0}" = "1" ] || [ "${PREPARE_OVERWRITE:-}" = "true" ]; then
  PREPARE_FLAGS+=(--overwrite)
fi

python3 -m examples.data_preprocess.prepare \
  --mode 'text' \
  --local_dir "${DATA_ROOT}" \
  --train_data_size "${train_data_size}" \
  --val_data_size "${val_data_size}" \
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

export VLLM_NCCL_SO_PATH="${VLLM_NCCL_SO_PATH:-/mnt/petrelfs/wurong/miniconda3/envs/verl-agent/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2}"
RAY_JOB_RUNTIME_ENV_JSON="$(python3 -c "import json, os; print(json.dumps({'excludes': ['logs', 'ray_log', 'swanlog'], 'env_vars': {'VLLM_NCCL_SO_PATH': os.environ.get('VLLM_NCCL_SO_PATH', '')}}))")"

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
    actor_rollout_ref.actor.trainable=True \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
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
    mem_adaptor.enable=true \
    mem_adaptor.use_actor_rollout_wg=true \
    mem_adaptor.train_memory_adaptor=false \
    mem_adaptor.model.path="${MODEL_PATH}" \
    env.env_name="${ENV_NAME}" \
    env.seed=0 \
    env.max_steps="${env_max_steps:-15}" \
    env.webshop.use_small="${WEBSHOP_USE_SMALL:-True}" \
    env.webshop.human_goals="${WEBSHOP_HUMAN_GOALS:-False}" \
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
    "${WEBSHOP_TRAIN_CLI[@]+"${WEBSHOP_TRAIN_CLI[@]}"}" \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='MemAdaptor_webshop' \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.default_local_dir="${TRAINER_CHECKPOINT_DIR}" \
    trainer.n_gpus_per_node="${trainer_n_gpus_per_node}" \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.validation_data_dir="${EXP_DIR}/val_traj" \
    trainer.val_before_train=False \
    trainer.val_only=False \
    "$@"
