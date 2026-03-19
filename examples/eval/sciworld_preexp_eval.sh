#!/usr/bin/env bash
set -x

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export VLLM_USE_MODELSCOPE=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=XFORMERS

GPU_NUM=4

source activate
conda activate agentgym-rl
export VLLM_ATTENTION_BACKEND=XFORMERS


TASK_NAME="sciworld"
ENV_SERVER_URL="http://0.0.0.0:12453"

EXPERIMENT_NAME="Qwen2.5-3B-Instruct-no_memory"
EXPERIMENTS_ROOT="exp_results/MemAdaptor/pre_exp"
DIR_DIR="/home/wurong/workspace/MemAdaptor/data/AgentGym"
MODEL_PATH="/nvme/public_models/Qwen2.5-3B-Instruct"

MEMORY_ENABLED=False

SAMPLE_NUM=1
MAX_ROUNDS=30
BATCH_SIZE=32

RETRIEVAL_MODE="agentic"
EMBEDDING_API_URL="http://10.140.37.68:8081/v1"


EXP_DIR="${EXPERIMENTS_ROOT}/${TASK_NAME}/${EXPERIMENT_NAME}"

ROLLOUT_LOG_DIR="${EXP_DIR}/logs"
MEMORY_STORE_DIR="${EXP_DIR}/memory_vdb"
RESULTS_DIR="${EXP_DIR}/results"

REBUILD_SOURCE_PATH="/home/wurong/workspace/MemAdaptor/data/AgentGym/AgentTraj-L/${TASK_NAME}_train_memory_records-gpt-5.1.jsonl"


# 创建所需目录
mkdir -p "${ROLLOUT_LOG_DIR}" "${RESULTS_DIR}"

echo "  根目录: ${EXP_DIR}"
echo "  日志:   ${ROLLOUT_LOG_DIR}"
echo "  数据库: ${MEMORY_STORE_DIR}"
echo "  结果:   ${RESULTS_DIR}"




HYDRA_FULL_ERROR=1 python3 -m verl.agent_trainer.main_generation \
    trainer.n_gpus_per_node="${GPU_NUM}" \
    data.path="${DIR_DIR}/eval" \
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
    rollout.temperature=0.7 \
    rollout.max_model_len=32768 \
    rollout.max_tokens=200 \
    rollout.tensor_model_parallel_size=1 \
    rollout.rollout_log_dir="${ROLLOUT_LOG_DIR}" \
    rollout.memory.enabled=${MEMORY_ENABLED} \
    rollout.memory.backend=milvus \
    rollout.memory.mode=init_if_missing \
    rollout.memory.write_back=False \
    rollout.memory.only_successful=True \
    rollout.memory.store_dir="${MEMORY_STORE_DIR}" \
    rollout.memory.clean_before_init=False \
    rollout.memory.rebuild_source_path="${REBUILD_SOURCE_PATH}" \
    rollout.memory.embedding_api_url="${EMBEDDING_API_URL}" \
    rollout.memory.retrieval_mode="${RETRIEVAL_MODE}"

status=$?
exit $status
