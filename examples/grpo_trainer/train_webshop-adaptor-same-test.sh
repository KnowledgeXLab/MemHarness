#!/usr/bin/env bash
#SBATCH --job-name=e05-web-adaptor-7b-random-state-test
#SBATCH --partition=DataFrontier_Explore
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --quotatype=reserved
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G 
#SBATCH --output=logs/mem_adaptor/webshop/adaptor_7b_random_state_test_%j.out
#SBATCH --error=logs/mem_adaptor/webshop/adaptor_7b_random_state_test_%j.err

set -x
set -euo pipefail

mkdir -p logs/mem_adaptor/webshop

ENGINE="vllm"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES 2>/dev/null || true
export HYDRA_FULL_ERROR=1
export WANDB_MODE="offline"

# 单一 checkpoint：主策略 rollout 与 MemAdaptor 共用（mem_adaptor.use_actor_rollout_wg=true）
# MODEL_PATH='models/save_models/mem_adaptor/cold_start/webshop/qwen2.5-7b-cold-start-20260511/global_step_125'
# MODEL_PATH='models/save_models/mem_adaptor/cold_start/webshop/qwen2.5-7b-cold-start-20260519/global_step_250'
# MODEL_PATH='models/save_models/mem_adaptor/webshop/train_adaptor-same-7B-cold_start_20260519_epoch1-with_agentic_memory-retrieve_memory_text-self_distill/global_step_115/actor-hf'
MODEL_PATH='models/save_models/mem_adaptor/webshop/train_adaptor-same-7B-cold_start_20260706_epoch2-with_agentic_memory-retrieve_memory_text-self_distill/global_step_80/actor/huggingface'
# MODEL_PATH='models/save_models/mem_adaptor/cold_start/webshop/qwen2.5-7b-cold-start-20260706/global_step_400'


trainer_n_gpus_per_node=2

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
  echo "[error] Submit from repo root: sbatch examples/grpo_trainer/train_webshop-adaptor-same.sh" >&2
  exit 1
fi
source "${SCRIPT_DIR}/memory_eval_helpers.sh"

REPO_DATA_DIR="$(resolve_repo_data_dir)" || exit 1
setup_verl_agent_text_data_paths webshop || exit 1

export WANDB_DIR='wandb_logs'

num_cpus_per_env_worker=0.1

TASK_NAME="webshop"

MEMORY_ENABLED="True"
MEMORY_WRITE_BACK="False"
EXPERIENCE_SUMMARIZER_MODE="self"
EXPERIENCE_SUMMARIZER_SCHEMA="compact"
RETRIEVAL_MODE="agentic"
RETRIEVE_KEY="memory_text"

# EMBEDDING_API_URL="http://10.140.37.18:8887/v1"
# EMBEDDING_API_KEY="DataFrontier_bge_m3"
EMBEDDING_API_URL="http://10.140.37.55:8081/v1"
EMBEDDING_API_KEY="DataFrontier_bge_m3"

USE_GENERAL_MODEL_RETRIEVAL_HINT="${USE_GENERAL_MODEL_RETRIEVAL_HINT:-1}"
RETRIEVAL_INSTRUCTION_PROMPT="${RETRIEVAL_INSTRUCTION_PROMPT:-}"
RETRIEVAL_INSTRUCTION_PROMPT_FILE="${RETRIEVAL_INSTRUCTION_PROMPT_FILE:-}"

MEMORY_REMOTE_SLURM="True"
MEMORY_REMOTE_PARTITION="p-cpu-new"  # DataFrontier_Explore / p-cpu-new
MEMORY_REMOTE_SERVER_PORT="8784"
MEMORY_REMOTE_EXCLUDE_NODES=''
MEMORY_APPTAINER_SIF="/mnt/petrelfs/wurong/glibc_ubuntu22.sif"
MEMORY_CONDA_SH="/mnt/petrelfs/wurong/miniconda3/etc/profile.d/conda.sh"
MEMORY_REMOTE_CONDA_ENV="verl-agent"

MEMORY_REBUILD_SOURCE_PATH=""

EXPERIENCE_UTILITY_ENABLE="True"
EXPERIENCE_UTILITY_PRUNE_EVERY_N_GLOBAL_STEPS=20
EXPERIENCE_UTILITY_PRUNE_SCORE_THRESHOLD=0.3
EXPERIENCE_UTILITY_MIN_USES_BEFORE_PRUNE=3

EXPERIMENT_NAME="train_adaptor-same-7b-cold_start_20260706_epoch2-step_80-random_state-test"
EXPERIMENTS_ROOT="data/MemAdaptor/exp_results"

if [ "${MEMORY_ENABLED}" = "True" ]; then
  EXPERIMENT_NAME="${EXPERIMENT_NAME}-with_${RETRIEVAL_MODE}_memory"
  EXPERIMENT_NAME="${EXPERIMENT_NAME}-retrieve_${RETRIEVE_KEY}"
  EXPERIMENT_NAME="${EXPERIMENT_NAME}-${EXPERIENCE_SUMMARIZER_MODE}_distill"
else
  EXPERIMENT_NAME="${EXPERIMENT_NAME}-no_memory"
fi

EXP_DIR="${EXPERIMENTS_ROOT}/${TASK_NAME}/${EXPERIMENT_NAME}"

# MEMORY_STORE_DIR="${EXP_DIR}/memory_vdb"
# MEMORY_STORE_DIR="data/MemAdaptor/exp_results/webshop/train_adaptor-same-7B-cold_start_20260519_epoch1-with_agentic_memory-retrieve_memory_text-self_distill/memory_vdb"
MEMORY_STORE_DIR="data/MemAdaptor/exp_results/webshop/train_adaptor-same-7B-cold_start_20260706_epoch2-with_agentic_memory-retrieve_memory_text-self_distill/memory_vdb"

TRAINER_CHECKPOINT_DIR="models/save_models/mem_adaptor/${TASK_NAME}/${EXPERIMENT_NAME}"

mkdir -p "${EXP_DIR}"
mkdir -p "${TRAINER_CHECKPOINT_DIR}"
LOG_FILE="${EXP_DIR}/train_webshop_adaptor_same_7b_random_state-$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "${LOG_FILE}") 2>&1
echo "[log] Writing full run output to: ${LOG_FILE}"
echo "[log] REPO_DATA_DIR=${REPO_DATA_DIR}"
echo "[log] DATA_ROOT=${DATA_ROOT}"
echo "[log] MODEL_PATH=${MODEL_PATH}"
echo "[log] trainer.default_local_dir=${TRAINER_CHECKPOINT_DIR}"

train_data_size=16
val_data_size=250
group_size=8

DATA_MAX_PROMPT_LENGTH=6144
DATA_MAX_RESPONSE_LENGTH=512
SUMMARIZER_MAX_PROMPT_TOKENS=12288
ROLLOUT_MAX_MODEL_LEN=$((SUMMARIZER_MAX_PROMPT_TOKENS + DATA_MAX_RESPONSE_LENGTH))
if [ -z "${ROLLOUT_MAX_MODEL_LEN}" ]; then
  ROLLOUT_MAX_MODEL_LEN=$((SUMMARIZER_MAX_PROMPT_TOKENS + DATA_MAX_RESPONSE_LENGTH))
fi

tensor_model_parallel_size=2

PREPARE_FLAGS=()
if [ "${PREPARE_OVERWRITE:-0}" = "1" ] || [ "${PREPARE_OVERWRITE:-}" = "true" ]; then
  PREPARE_FLAGS+=(--overwrite)
fi

python3 -m examples.data_preprocess.prepare \
  --mode 'text' \
  --local_dir "${DATA_ROOT}" \
  --infer_webshop_sizes \
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

append_retrieval_instruction_cli

export VLLM_NCCL_SO_PATH=/mnt/petrelfs/wurong/miniconda3/envs/verl-agent-webshop/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2
RAY_JOB_RUNTIME_ENV_JSON="$(python3 -c "import json, os; print(json.dumps({'excludes': ['logs', 'ray_log', 'swanlog'], 'env_vars': {'VLLM_NCCL_SO_PATH': os.environ['VLLM_NCCL_SO_PATH']}}))")"

if [ -z "${VLLM_NCCL_SO_PATH:-}" ]; then
  VLLM_NCCL_SO_PATH="$(
    python3 -c 'import pathlib,sys; lib=pathlib.Path(sys.prefix)/"lib"; xs=sorted(lib.glob("python*/site-packages/nvidia/nccl/lib/libnccl.so.2")); print(xs[0] if xs else "")' 2>/dev/null || true
  )"
  export VLLM_NCCL_SO_PATH
fi
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
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.actor.trainable=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${tensor_model_parallel_size}" \
  actor_rollout_ref.rollout.max_model_len="${ROLLOUT_MAX_MODEL_LEN}" \
  actor_rollout_ref.rollout.name="${ENGINE}" \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.use_invalid_action_penalty=True \
  actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
  algorithm.use_kl_in_reward=False \
  mem_adaptor.enable=True \
  mem_adaptor.use_actor_rollout_wg=True \
  mem_adaptor.train_memory_adaptor=false \
  mem_adaptor.retrieved_state_mode=random_vdb \
  mem_adaptor.model.path="${MODEL_PATH}" \
  env.env_name=Webshop \
  env.seed=0 \
  env.max_steps=15 \
  env.memory.enabled="${MEMORY_ENABLED}" \
  env.memory.store_dir="${MEMORY_STORE_DIR}" \
  env.memory.write_back="${MEMORY_WRITE_BACK}" \
  env.memory.experience_summarizer.mode="${EXPERIENCE_SUMMARIZER_MODE}" \
  env.memory.experience_summarizer.schema="${EXPERIENCE_SUMMARIZER_SCHEMA}" \
  env.memory.experience_summarizer.summarizer_max_prompt_tokens="${SUMMARIZER_MAX_PROMPT_TOKENS}" \
  env.memory.experience_summarizer.summarizer_trajectory_min_turns_kept=4 \
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
  trainer.project_name='MemAdaptor_webshop' \
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
  trainer.val_before_train=True \
  trainer.val_only=True \
  "$@"
