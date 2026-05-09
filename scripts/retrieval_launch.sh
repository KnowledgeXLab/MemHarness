#!/bin/bash
# --- Job Configuration ---
# Define the conda environment you want to use for this job.
CONDA_ENV_NAME="retriever"

# --- Script Paths ---
# The absolute path to the setup script we created.
# This path is viewed from *inside* the container, thanks to the --bind mount.
SETUP_SCRIPT_PATH="/mnt/petrelfs/wurong/workspace/Exp_RL/src/setup_conda_env.sh"
CONTAINER_IMAGE_PATH="/mnt/petrelfs/wurong/glibc_ubuntu22.sif"

# export CUDA_VISIBLE_DEVICES=0
echo "Starting retrieval server..."
# echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# 保存服务器信息供其他脚本访问
HOSTNAME=$(hostname)
IP=$(hostname -I | awk '{print $1}')
RETRIEVER_INFO_FILE="retriever_server_info.txt"

echo "Retriever服务器将在节点 $HOSTNAME ($IP) 上启动"

cat > $RETRIEVER_INFO_FILE << EOF
# Retriever服务器信息
# 生成时间: $(date)
RETRIEVER_HOST=$HOSTNAME
RETRIEVER_IP=$IP
RETRIEVER_PORT=8000
RETRIEVER_URL=http://$IP:8000
EOF

echo "服务器信息已保存到: $RETRIEVER_INFO_FILE"

file_path=data/exp-rl/Wiki-corpus-embedd/
index_file=$file_path/e5_Flat.index
corpus_file=$file_path/wiki-18.jsonl
retriever_name=e5
retriever_path=models/intfloat/e5-base-v2


apptainer exec --nv --bind /mnt:/mnt ${CONTAINER_IMAGE_PATH} \
    bash -c "source ${SETUP_SCRIPT_PATH} ${CONDA_ENV_NAME} && python examples/search/retriever/retrieval_server.py \
                                            --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk 3 \
                                            --retriever_name $retriever_name \
                                            --retriever_model $retriever_path \
                                            --faiss_gpu"

