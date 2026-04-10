#!/usr/bin/env bash
set -x
set -euo pipefail

ENGINE="vllm" # 须为 vllm（或 hf）；勿用 openai_api 搭配 train_memory_adaptor, vllm | openai_api；openai_api 时需 export OPENAI_API_KEY
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export CUDA_VISIBLE_DEVICES=0,1
export HYDRA_FULL_ERROR=1
export WANDB_MODE="offline"

# global_pool（Reasoning / Ref / Critic 等）每节点 GPU 数
trainer_n_gpus_per_node=6
# mem_adaptor 专用池每节点 GPU 数（main_ppo 默认与 nnodes 同长度的 1 列表）
mem_adaptor_gpus_per_node=2
GPU_NUM="${trainer_n_gpus_per_node}"
# Ray Object Store（Plasma）内存上限，单位 GiB；传给 ray_init.object_store_memory（字节）
RAY_OBJECT_STORE_GIB=160
RAY_OBJECT_STORE_BYTES=$((RAY_OBJECT_STORE_GIB * 1024 * 1024 * 1024))

DATA_ROOT="data/verl-agent"
TRAIN_FILE="${DATA_ROOT}/text/train.parquet"
TEST_FILE="${DATA_ROOT}/text/test.parquet"
VAL_FILE="${TEST_FILE}"

num_cpus_per_env_worker=0.1

TASK_NAME="alfworld"

MEMORY_ENABLED=True
RETRIEVAL_MODE="agentic" # agentic | fixed
RETRIEVE_KEY="memory_text" # memory_text | state_text
EMBEDDING_API_URL=""
# 与 eval 一致：预构建记忆 jsonl；留空则不传参（沿用 yaml 默认 null）
MEMORY_REBUILD_SOURCE_PATH=""

EXPERIMENT_NAME="Qwen2.5-0.5B-grpo-train-smoke"
EXPERIMENTS_ROOT="data/exp_results/MemAdaptor/pre_exp_train"
MODEL_PATH="/nvme/public_models/Qwen2.5-0.5B-Instruct"
# Adaptor 初始化权重（默认与 Reasoning 同路径；可换小模型或 LoRA 基底）
MEM_ADAPTOR_MODEL_PATH="/nvme/public_models/Qwen2.5-0.5B-Instruct"

if [ "${MEMORY_ENABLED}" = "True" ]; then
  EXPERIMENT_NAME="${EXPERIMENT_NAME}-with_${RETRIEVAL_MODE}_memory"
  EXPERIMENT_NAME="${EXPERIMENT_NAME}-retrieve_${RETRIEVE_KEY}"
else
  EXPERIMENT_NAME="${EXPERIMENT_NAME}-no_memory"
fi
EXPERIMENT_NAME="${EXPERIMENT_NAME}-frozen_reasoning_train_mem_adaptor"

EXP_DIR="${EXPERIMENTS_ROOT}/${TASK_NAME}/${EXPERIMENT_NAME}"
MEMORY_STORE_DIR="${EXP_DIR}/memory_vdb"

mkdir -p "${EXP_DIR}"
LOG_FILE="${EXP_DIR}/train_alfworld-$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "${LOG_FILE}") 2>&1
echo "[log] Writing full run output to: ${LOG_FILE}"

# 训练 batch：须与 env.rollout.n（GRPO group）及数据量匹配；可先改小做冒烟
train_data_size="${train_data_size:-16}"
val_data_size="${val_data_size:-128}"
group_size="${group_size:-8}"
max_concurrent="${max_concurrent:-32}"

# TP 与 GPU 数一致（单卡训练请保持1）
tensor_model_parallel_size="${tensor_model_parallel_size:-${GPU_NUM}}"

VALIDATE_ON_TRAIN_SPLIT="${VALIDATE_ON_TRAIN_SPLIT:-False}"

PREPARE_FLAGS=()
if [ "${PREPARE_OVERWRITE:-0}" = "1" ] || [ "${PREPARE_OVERWRITE:-}" = "true" ]; then
  PREPARE_FLAGS+=(--overwrite)
fi

python3 -m examples.data_preprocess.prepare \
  --mode 'text' \
  --local_dir "${DATA_ROOT}" \
  --infer_alfworld_sizes \
  --alfworld_eval_split eval_in_distribution \
  "${PREPARE_FLAGS[@]}"

MEMORY_CLI=()
if [ -n "${MEMORY_REBUILD_SOURCE_PATH}" ]; then
  MEMORY_CLI+=(env.memory.rebuild_source_path="${MEMORY_REBUILD_SOURCE_PATH}")
fi
if [ -n "${EMBEDDING_API_URL}" ]; then
  MEMORY_CLI+=(env.memory.embedding_api_url="${EMBEDDING_API_URL}")
fi

python3 -m verl.trainer.main_ppo \
  ray_init.object_store_memory="${RAY_OBJECT_STORE_BYTES}" \
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
  actor_rollout_ref.actor.trainable=false \
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
  env.env_name=alfworld/AlfredTWEnv \
  env.alfworld.validate_on_train_split="${VALIDATE_ON_TRAIN_SPLIT}" \
  env.seed=0 \
  env.max_steps=30 \
  env.memory.enabled="${MEMORY_ENABLED}" \
  env.memory.store_dir="${MEMORY_STORE_DIR}" \
  env.memory.retrieval_mode="${RETRIEVAL_MODE}" \
  env.memory.retrieve_key="${RETRIEVE_KEY}" \
  "${MEMORY_CLI[@]}" \
  env.rollout.n="${group_size}" \
  env.resources_per_worker.num_cpus="${num_cpus_per_env_worker}" \
  trainer.critic_warmup=0 \
  trainer.logger=['console'] \
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
