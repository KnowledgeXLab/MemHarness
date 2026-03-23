#!/usr/bin/env bash
# 需用 bash 运行（含进程替换 `>(tee ...)`）；推荐: bash eval_alfworld-preexp.sh
set -x
set -euo pipefail
ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export CUDA_VISIBLE_DEVICES=0,1
export HYDRA_FULL_ERROR=1
export WANDB_MODE=offline

GPU_NUM=2
# Ray Object Store（Plasma）内存上限，单位 GiB；传给 ray_init.object_store_memory（字节）
RAY_OBJECT_STORE_GIB=96
RAY_OBJECT_STORE_BYTES=$((RAY_OBJECT_STORE_GIB * 1024 * 1024 * 1024))

DATA_ROOT="data/verl-agent"
TRAIN_FILE="${DATA_ROOT}/text/train.parquet"
TEST_FILE="${DATA_ROOT}/text/test.parquet"

# val_only 时只会跑 data.val_files 指向的 parquet：
#   EVAL_ON_SPLIT=test（默认） -> test.parquet；val 环境为 eval 划分
#   EVAL_ON_SPLIT=train        -> train.parquet 全量；须 VALIDATE_ON_TRAIN_SPLIT=True
# 示例：EVAL_ON_SPLIT=train bash examples/grpo_trainer/eval_alfworld-preexp.sh
EVAL_ON_SPLIT="train"  # train | test
VALIDATE_ON_TRAIN_SPLIT=True
if [ "${EVAL_ON_SPLIT}" = "train" ]; then
  VAL_FILE="${TRAIN_FILE}"
  VALIDATION_TRAJ_SUBDIR="train_traj"
  VALIDATE_ON_TRAIN_SPLIT=True
else
  VAL_FILE="${TEST_FILE}"
  VALIDATION_TRAJ_SUBDIR="val_traj"
fi

num_cpus_per_env_worker=0.1 # The CPU resource allocated for each environment worker. If you want to use less CPU resources, you can decrease this value.

TASK_NAME="alfworld"

MEMORY_ENABLED=True
RETRIEVAL_MODE="agentic"    # agentic | fixed
RETRIEVE_KEY="memory_text"  # memory_text | state_text
EMBEDDING_API_URL="http://10.140.37.35:8081/v1"
MEMORY_REBUILD_SOURCE_PATH="/home/wurong/workspace/MemAdaptor/data/AgentGym/AgentTraj-L/${TASK_NAME}_train_memory_records-gpt-5.1.jsonl"


EXPERIMENT_NAME="Qwen2.5-1.5B-Instruct"
EXPERIMENTS_ROOT="data/exp_results/MemAdaptor/pre_exp"
MODEL_PATH="/nvme/public_models/Qwen2.5-1.5B-Instruct"

if [ "${MEMORY_ENABLED}" == "True" ]; then
    EXPERIMENT_NAME="${EXPERIMENT_NAME}-with_${RETRIEVAL_MODE}_memory"
    EXPERIMENT_NAME="${EXPERIMENT_NAME}-retrieve_${RETRIEVE_KEY}"
else
    EXPERIMENT_NAME="${EXPERIMENT_NAME}-no_memory"
fi


EXP_DIR="${EXPERIMENTS_ROOT}/${TASK_NAME}/${EXPERIMENT_NAME}"
MEMORY_STORE_DIR="${EXP_DIR}/memory_vdb"

mkdir -p "${EXP_DIR}"
LOG_FILE="${EXP_DIR}/eval_alfworld-$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "${LOG_FILE}") 2>&1
echo "[log] Writing full run output to: ${LOG_FILE}"

train_data_size=16
# 验证并行环境数 = AlfWorld val 并行 worker 数。若样本数不能整除 batch，须设 VAL_DROP_LAST=true（会丢掉最后不足一批，最多 val_batch_size-1 条）
# 例：3553 训练集 + batch 8 -> 少评 1 条；或改用能整除的 batch（如 11/17/19）
val_batch_size=4
VAL_DROP_LAST=True
group_size=8

# 默认：从 AlfredTWEnv 推断 train/eval 全量可玩 game 数并生成占位 parquet（与轨迹条数一致）
# 首次或需改条数：PREPARE_OVERWRITE=1 bash examples/grpo_trainer/eval_alfworld-preexp.sh
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

python3 -m verl.trainer.main_ppo \
    ray_init.object_store_memory=${RAY_OBJECT_STORE_BYTES} \
    algorithm.adv_estimator=grpo \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${VAL_FILE} \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_batch_size \
    data.val_drop_last=${VAL_DROP_LAST} \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
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
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
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
    env.env_name=alfworld/AlfredTWEnv \
    env.alfworld.validate_on_train_split=${VALIDATE_ON_TRAIN_SPLIT} \
    env.seed=0 \
    env.max_steps=50 \
    env.memory.enabled=${MEMORY_ENABLED} \
    env.memory.store_dir=${MEMORY_STORE_DIR} \
    env.memory.rebuild_source_path=${MEMORY_REBUILD_SOURCE_PATH} \
    env.memory.embedding_api_url=${EMBEDDING_API_URL} \
    env.memory.retrieval_mode=${RETRIEVAL_MODE} \
    env.memory.retrieve_key=${RETRIEVE_KEY} \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='MemAdaptor_alfworld' \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${GPU_NUM} \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.validation_data_dir=${EXP_DIR}/${VALIDATION_TRAJ_SUBDIR} \
    trainer.val_before_train=True \
    trainer.val_only=True $@
