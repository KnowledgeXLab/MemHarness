#!/usr/bin/env bash
set -x

export CUDA_VISIBLE_DEVICES="0,1"
export VLLM_USE_MODELSCOPE=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=XFORMERS

TASK_NAME="sciworld"
ENV_SERVER_URL=""

EXPERIMENT_NAME="Qwen3.5-2B-"
EXPERIMENTS_ROOT="exp_results/MemAdaptor/pre_exp"


MODEL_PATH="/nvme/public_models/Qwen3.5-2B"

SAMPLE_NUM=1
MAX_ROUNDS=30
BATCH_SIZE=32

RETRIEVAL_MODE="agentic"
EMBEDDING_API_URL="http://10.140.37.68:8081/v1"


EXP_DIR="${EXPERIMENTS_ROOT}/${EXPERIMENT_NAME}"

ROLLOUT_LOG_DIR="${EXP_DIR}/logs"
MEMORY_STORE_DIR="${EXP_DIR}/memory_vdb"
RESULTS_DIR="${EXP_DIR}/results"

MEMORY_DB_PATH="data/AgentGym/AgentTraj-L/alfworld_train_memory_records-gpt-5.1.jsonl"


# 创建所需目录
mkdir -p "${ROLLOUT_LOG_DIR}" "${RESULTS_DIR}"

echo "  根目录: ${EXP_DIR}"
echo "  日志:   ${ROLLOUT_LOG_DIR}"
echo "  数据库: ${MEMORY_STORE_DIR}"
echo "  结果:   ${RESULTS_DIR}"


cd AgentGym-RL

if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate agentgym-rl
fi

HYDRA_FULL_ERROR=1 python3 -m verl.agent_trainer.main_generation \
    data.path="AgentEval/${TASK_NAME}" \
    data.max_prompt_length=1024 \
    data.max_response_length=8192 \
    data.n_samples="${SAMPLE_NUM}" \
    data.batch_size="${BATCH_SIZE}" \
    agentgym.task_name="${TASK_NAME}" \
    agentgym.env_addr="${ENV_SERVER_URL}" \
    agentgym.max_rounds="${MAX_ROUNDS}" \
    agentgym.timeout=500 \
    model.path="${MODEL_PATH}" \
    rollout.gpu_memory_utilization=0.95 \
    rollout.temperature=1 \
    rollout.max_model_len=32768 \
    rollout.max_tokens=200 \
    rollout.tensor_model_parallel_size=1 \
    rollout.rollout_log_dir="${ROLLOUT_LOG_DIR}" \
    rollout.memory.enabled=True \
    rollout.memory.backend=milvus \
    rollout.memory.mode=init_if_missing \
    rollout.memory.write_back=False \
    rollout.memory.only_successful=True \
    rollout.memory.store_dir="${MEMORY_STORE_DIR}" \
    rollout.memory.milvus_db_path="${MEMORY_DB_PATH}" \
    rollout.memory.clean_before_init=False \
    rollout.memory.embedding_api_url="${EMBEDDING_API_URL}" \
    rollout.memory.retrieval_mode="${RETRIEVAL_MODE}"

status=$?
exit $status
